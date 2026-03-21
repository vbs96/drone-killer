#!/usr/bin/env python3
"""Convert the SSD MobileNet frozen graph to TFLite WITHOUT Flex ops.

Uses the frozen_inference_graph.pb directly, feeding the preprocessed
image (after resize/normalize) and extracting raw box/class predictions
from the BoxPredictor concat nodes — bypassing both the Preprocessor
and Postprocessor that use TensorArray ops.

NMS and box decoding are done in C++ at inference time.

Usage:
    source .venv/bin/activate
    python3 convert_model.py
"""

import numpy as np
import tensorflow as tf

FROZEN_GRAPH = "model/frozen_inference_graph.pb"
OUTPUT_PATH = "model/drone_detect.tflite"
ANCHORS_PATH = "model/anchors.bin"  # raw float32 [1917, 4]

# ── 1. Strip Preprocessor/Postprocessor from the frozen graph ─────────
# The Preprocessor uses TensorArray ops (for batch resizing) and the
# Postprocessor uses them for NMS.  TFLite can't handle these.
# We prune the graph to keep ONLY nodes reachable from our desired
# outputs (concat, concat_1) and replace the preprocessor output
# with a new Placeholder.

graph_def = tf.compat.v1.GraphDef()
with open(FROZEN_GRAPH, "rb") as f:
    graph_def.ParseFromString(f.read())

PREPROCESSOR_OUTPUT = "Preprocessor/map/TensorArrayStack/TensorArrayGatherV3"
NEW_INPUT_NAME = "preprocessed_input"
DESIRED_OUTPUTS = ["concat", "concat_1"]

# Build name→node map
node_map = {node.name: node for node in graph_def.node}

# BFS from desired outputs to find all required nodes
required = set()
queue = list(DESIRED_OUTPUTS)
while queue:
    name = queue.pop()
    if name in required or name == PREPROCESSOR_OUTPUT:
        continue
    required.add(name)
    if name in node_map:
        for inp in node_map[name].input:
            # Strip control dependency prefix
            dep = inp.lstrip("^").split(":")[0]
            queue.append(dep)

# Build new graph with only required nodes
new_graph_def = tf.compat.v1.GraphDef()
for node in graph_def.node:
    if node.name not in required:
        continue
    new_node = new_graph_def.node.add()
    new_node.CopyFrom(node)
    for i, inp in enumerate(new_node.input):
        dep = inp.lstrip("^").split(":")[0]
        if dep == PREPROCESSOR_OUTPUT:
            new_node.input[i] = NEW_INPUT_NAME

# Add new placeholder input
placeholder = new_graph_def.node.add()
placeholder.op = "Placeholder"
placeholder.name = NEW_INPUT_NAME
placeholder.attr["dtype"].type = 1  # DT_FLOAT
placeholder.attr["shape"].shape.dim.add().size = 1
placeholder.attr["shape"].shape.dim.add().size = 300
placeholder.attr["shape"].shape.dim.add().size = 300
placeholder.attr["shape"].shape.dim.add().size = 3

# Save pruned graph
pruned_path = "model/frozen_pruned.pb"
with open(pruned_path, "wb") as f:
    f.write(new_graph_def.SerializeToString())
print(f"Saved pruned graph: {pruned_path}")

# ── 2. Convert pruned graph to TFLite ─────────────────────────────────
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    pruned_path,
    input_arrays=[NEW_INPUT_NAME],
    output_arrays=["concat", "concat_1"],
    input_shapes={NEW_INPUT_NAME: [1, 300, 300, 3]},
)
tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)
print(f"Saved {OUTPUT_PATH}  ({len(tflite_model) / 1e6:.1f} MB)")

# ── 2. Generate anchor boxes ─────────────────────────────────────────
# SSD MobileNet v1 300x300, 6 feature map layers.
# From pipeline.config: min_scale=0.2, max_scale=0.95,
# aspect_ratios=[1.0, 2.0, 0.5, 3.0, 0.333]

def generate_anchors():
    feature_map_sizes = [19, 10, 5, 3, 2, 1]
    num_layers = 6
    min_scale = 0.2
    max_scale = 0.95
    aspect_ratios_per_layer = [
        [1.0, 2.0, 0.5],
        [1.0, 2.0, 0.5, 3.0, 1/3.0],
        [1.0, 2.0, 0.5, 3.0, 1/3.0],
        [1.0, 2.0, 0.5, 3.0, 1/3.0],
        [1.0, 2.0, 0.5, 3.0, 1/3.0],
        [1.0, 2.0, 0.5, 3.0, 1/3.0],
    ]
    scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
              for i in range(num_layers)]
    scales.append(1.0)

    anchors = []
    for layer_idx, fm_size in enumerate(feature_map_sizes):
        scale = scales[layer_idx]
        scale_next = scales[layer_idx + 1]
        ars = aspect_ratios_per_layer[layer_idx]
        # reduce_boxes_in_lowest_layer: no geometric-mean anchor on layer 0
        reduce = (layer_idx == 0)

        for y in range(fm_size):
            for x in range(fm_size):
                cy = (y + 0.5) / fm_size
                cx = (x + 0.5) / fm_size

                # aspect_ratio=1 at current scale
                anchors.append([cy, cx, scale, scale])
                # aspect_ratio=1 at geometric mean of current+next scale
                if not reduce:
                    extra = np.sqrt(scale * scale_next)
                    anchors.append([cy, cx, extra, extra])

                for ar in ars:
                    if ar == 1.0:
                        continue
                    w = scale * np.sqrt(ar)
                    h = scale / np.sqrt(ar)
                    anchors.append([cy, cx, h, w])

    return np.array(anchors, dtype=np.float32)

anchors = generate_anchors()
print(f"Generated {anchors.shape[0]} anchors")
assert anchors.shape == (1917, 4), f"Expected 1917 anchors, got {anchors.shape[0]}"

anchors.tofile(ANCHORS_PATH)
print(f"Saved {ANCHORS_PATH}")
