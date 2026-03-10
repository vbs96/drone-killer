#!/usr/bin/env python3
import argparse
import json
from collections import deque
from typing import List

import numpy as np

from optimum.onnxruntime import ORTModelForAudioClassification
from transformers import AutoFeatureExtractor, pipeline


def percentile_aggregate(scores: List[float], q: float = 90.0) -> float:
    if not scores:
        return 0.0
    return float(np.percentile(np.array(scores, dtype=np.float32), q))


def find_drone_label(ort_model) -> str:
    id2label = ort_model.config.id2label
    labels = [id2label[i] for i in sorted(id2label.keys())]

    exact = [l for l in labels if l.strip().lower() == "drone"]
    if exact:
        return exact[0]

    contains = [l for l in labels if "drone" in l.lower()]
    if contains:
        return contains[0]

    raise RuntimeError(f"Could not find a 'drone' label in model labels: {labels}")


def score_chunk(clf, chunk: np.ndarray, sr: int, drone_label: str) -> float:
    preds = clf({"array": chunk, "sampling_rate": sr})
    for p in preds:
        if p["label"] == drone_label:
            return float(p["score"])
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-fifo", required=True)
    ap.add_argument("--onnx_dir", default="onnx_drone_model")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--win", type=float, default=5.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--history", type=int, default=5)
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.onnx_dir)
    ort_model = ORTModelForAudioClassification.from_pretrained(args.onnx_dir)

    clf = pipeline(
        task="audio-classification",
        model=ort_model,
        feature_extractor=feature_extractor,
        top_k=None,
    )

    drone_label = find_drone_label(ort_model)

    win_samples = int(args.win * args.sr)
    hop_samples = int(args.hop * args.sr)
    chunk_samples = int(args.sr * (args.chunk_ms / 1000.0))
    chunk_bytes = chunk_samples * 4  # float32

    ring_buffer = np.zeros(win_samples, dtype=np.float32)
    filled = 0
    since_last_infer = 0
    stream_time_s = 0.0
    recent_scores = deque(maxlen=args.history)

    out_file = open(args.out, "a", encoding="utf-8") if args.out else None

    print(f"Listening on FIFO: {args.input_fifo}")
    print(f"Using label: {drone_label}")

    try:
        with open(args.input_fifo, "rb") as f:
            while True:
                raw = f.read(chunk_bytes)
                print(f"read bytes: {len(raw)}")
                if not raw or len(raw) < chunk_bytes:
                    continue

                chunk = np.frombuffer(raw, dtype=np.float32)
                n = len(chunk)
                stream_time_s += n / args.sr

                if n >= win_samples:
                    ring_buffer[:] = chunk[-win_samples:]
                    filled = win_samples
                else:
                    ring_buffer[:-n] = ring_buffer[n:]
                    ring_buffer[-n:] = chunk
                    filled = min(win_samples, filled + n)

                since_last_infer += n

                if filled < win_samples or since_last_infer < hop_samples:
                    continue

                since_last_infer = 0

                raw_score = score_chunk(clf, ring_buffer, args.sr, drone_label)
                recent_scores.append(raw_score)
                smoothed_score = percentile_aggregate(list(recent_scores), q=90.0)
                decision = "drone" if smoothed_score >= args.threshold else "no_drone"

                event = {
                    "t_end": round(stream_time_s, 2),
                    "t_start": round(stream_time_s - args.win, 2),
                    "raw_score": raw_score,
                    "smoothed_score": smoothed_score,
                    "decision": decision,
                }

                if out_file:
                    out_file.write(json.dumps(event) + "\n")
                    out_file.flush()

                marker = "🚨" if decision == "drone" else "·"
                changed = decision != prev_decision
                change_tag = " CHANGE" if changed else ""

                print(
                    f"{marker}{change_tag} "
                    f"window=({event['t_start']:.2f}s-{event['t_end']:.2f}s) "
                    f"raw={raw_score:.3f} smooth={smoothed_score:.3f} "
                    f"decision={decision}"
                )

    finally:
        if out_file:
            out_file.close()


if __name__ == "__main__":
    main()