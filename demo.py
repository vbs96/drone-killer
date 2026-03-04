import re
import time
import tensorflow as tf
import cv2
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────
PATH_TO_MODEL_DIR = 'PaperBasedANNModels/droneInfGraph401092'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + '/saved_model'
PATH_TO_LABELS = PATH_TO_MODEL_DIR + '/object-detection.pbtxt'
MIN_SCORE_THRESH = 0.5   # Only show detections above this confidence


# ── Parse label map (.pbtxt) without the object_detection package ─────
def parse_labelmap(path):
    """Parse a .pbtxt label map file and return {id: name} dict."""
    with open(path, 'r') as f:
        text = f.read()
    category_index = {}
    for item in re.findall(r'item\s*\{(.*?)\}', text, re.DOTALL):
        id_match = re.search(r'id\s*:\s*(\d+)', item)
        name_match = re.search(r"name\s*:\s*['\"](.+?)['\"]", item)
        if id_match and name_match:
            category_index[int(id_match.group(1))] = name_match.group(1)
    return category_index


# ── Load model ─────────────────────────────────────────────────────────
print('Loading model...')
model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
detect_fn = model.signatures['serving_default']
print('Model loaded.')

# Label map
category_index = parse_labelmap(PATH_TO_LABELS)
print(f'Labels: {category_index}')

# ── Run inference on an image ──────────────────────────────────────────
image_path = 'heavy-lift-drone.jpg'
image_np = cv2.imread(image_path)
if image_np is None:
    raise FileNotFoundError(f'Could not read image: {image_path}')

# The model expects RGB uint8; OpenCV loads BGR
image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
input_tensor = tf.convert_to_tensor(np.expand_dims(image_rgb, 0), dtype=tf.uint8)

t_start = time.perf_counter()
detections = detect_fn(inputs=input_tensor)
t_end = time.perf_counter()

elapsed_ms = (t_end - t_start) * 1000
h, w = image_np.shape[:2]
megapixels = h * w / 1e6
ms_per_mp = elapsed_ms / megapixels
print(f'Inference: {elapsed_ms:.1f} ms  |  Image: {megapixels:.2f} MP  |  {ms_per_mp:.1f} ms/MP')

# Extract results
boxes = detections['detection_boxes'][0].numpy()      # [N, 4] normalised
classes = detections['detection_classes'][0].numpy().astype(int)  # [N]
scores = detections['detection_scores'][0].numpy()     # [N]

# ── Draw bounding boxes ───────────────────────────────────────────────
h, w = image_np.shape[:2]
for box, cls, score in zip(boxes, classes, scores):
    if score < MIN_SCORE_THRESH:
        continue
    ymin, xmin, ymax, xmax = box
    left, top, right, bottom = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)

    label = category_index.get(cls, f'class {cls}')
    text = f'{label}: {score:.0%}'

    cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 3)
    cv2.putText(image_np, text, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# ── Save result ────────────────────────────────────────────────────────
cv2.imwrite('output.jpg', image_np)
print(f'Saved output.jpg with {(scores >= MIN_SCORE_THRESH).sum()} detection(s).')