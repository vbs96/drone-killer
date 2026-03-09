#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import librosa

# ONNX-accelerated HF pipeline
from optimum.onnxruntime import ORTModelForAudioClassification
from transformers import AutoFeatureExtractor, pipeline


@dataclass
class WindowResult:
    t_start: float
    t_end: float
    score: float
    label: str


def load_audio_mono(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Loads audio (wav/mp3/etc.) via librosa (uses ffmpeg if installed).
    Returns mono float32 waveform in [-1, 1], resampled to target_sr.
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError("Empty audio.")
    # Prevent NaNs
    y = np.nan_to_num(y).astype(np.float32)
    return y, target_sr


def sliding_windows(y: np.ndarray, sr: int, win_s: float, hop_s: float) -> List[Tuple[int, int, float, float]]:
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    n = len(y)

    if n < win:
        # pad short audio
        pad = win - n
        y = np.pad(y, (0, pad), mode="constant")
        n = len(y)

    windows = []
    for start in range(0, n - win + 1, hop):
        end = start + win
        windows.append((start, end, start / sr, end / sr))
    return windows


def percentile_aggregate(scores: List[float], q: float = 90.0) -> float:
    if not scores:
        return 0.0
    return float(np.percentile(np.array(scores, dtype=np.float32), q))


def main():
    ap = argparse.ArgumentParser(description="Drone audio detection (file-based) using ONNX Runtime on Raspberry Pi.")
    ap.add_argument("audio_path", help="Path to audio file (wav/mp3/flac/etc.)")
    ap.add_argument("--onnx_dir", default="onnx_drone_model", help="Directory containing exported ONNX model")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    ap.add_argument("--win", type=float, default=5.0, help="Window size in seconds")
    ap.add_argument("--hop", type=float, default=1.0, help="Hop size in seconds")
    ap.add_argument("--threshold", type=float, default=0.7, help="File-level decision threshold")
    ap.add_argument("--topk", type=int, default=5, help="How many hottest windows to report")
    ap.add_argument("--out", default="result.json", help="Output JSON path")
    args = ap.parse_args()

    y, sr = load_audio_mono(args.audio_path, args.sr)

    # Load ONNX model + feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.onnx_dir)
    ort_model = ORTModelForAudioClassification.from_pretrained(args.onnx_dir)

    clf = pipeline(
        task="audio-classification",
        model=ort_model,
        feature_extractor=feature_extractor,
        top_k=None,  # we want full label distribution for safety; we'll pick "drone" label
    )

    # Determine which label corresponds to "drone"
    # We'll infer it from id2label, preferring an exact match.
    id2label = ort_model.config.id2label
    labels = [id2label[i] for i in sorted(id2label.keys())]
    drone_label_candidates = [l for l in labels if l.strip().lower() == "drone"]
    drone_label = drone_label_candidates[0] if drone_label_candidates else None

    if drone_label is None:
        # Fallback: try contains
        contains = [l for l in labels if "drone" in l.lower()]
        if contains:
            drone_label = contains[0]
        else:
            raise RuntimeError(f"Could not find a 'drone' label in model labels: {labels}")

    win_specs = sliding_windows(y, sr, args.win, args.hop)

    results: List[WindowResult] = []
    scores: List[float] = []

    for s, e, t0, t1 in win_specs:
        chunk = y[s:e]
        preds = clf({"array": chunk, "sampling_rate": sr})

        # preds is list of dicts: [{"label": "...", "score": ...}, ...]
        score = 0.0
        for p in preds:
            if p["label"] == drone_label:
                score = float(p["score"])
                break

        results.append(WindowResult(t_start=t0, t_end=t1, score=score, label=drone_label))
        scores.append(score)

    file_score = percentile_aggregate(scores, q=90.0)
    decision = "drone" if file_score >= args.threshold else "no_drone"

    # Top triggering windows
    hottest = sorted(results, key=lambda r: r.score, reverse=True)[: args.topk]

    out: Dict[str, Any] = {
        "audio_path": args.audio_path,
        "model": "preszzz/drone-audio-detection-05-17-trial-0 (ONNX Runtime)",
        "sample_rate": sr,
        "window_seconds": args.win,
        "hop_seconds": args.hop,
        "drone_label": drone_label,
        "file_score_p90": file_score,
        "threshold": args.threshold,
        "decision": decision,
        "top_windows": [
            {"t_start": r.t_start, "t_end": r.t_end, "score": r.score}
            for r in hottest
        ],
        "all_window_scores": [
            {"t_start": r.t_start, "t_end": r.t_end, "score": r.score}
            for r in results
        ],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(
        {
            "decision": decision,
            "file_score_p90": file_score,
            "drone_label": drone_label,
            "top_windows": out["top_windows"],
            "saved_to": args.out
        },
        indent=2
    ))


if __name__ == "__main__":
    main()