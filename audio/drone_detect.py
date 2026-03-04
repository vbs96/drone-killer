#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import librosa
import soundfile as sf

from transformers import pipeline


@dataclass
class WindowResult:
    t_start: float
    t_end: float
    score: float
    label: str


def load_audio_mono(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Loads audio as mono float32 in [-1, 1] and resamples to target_sr.
    librosa will use ffmpeg for many formats if ffmpeg is installed.
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError("Empty audio file.")
    y = np.nan_to_num(y).astype(np.float32)
    return y, target_sr


def sliding_windows(n_samples: int, sr: int, win_s: float, hop_s: float) -> List[Tuple[int, int, float, float]]:
    """
    Returns [(start_idx, end_idx, t0, t1), ...] for overlapping windows.
    Pads last segment by ignoring remainder (we pad audio separately if needed).
    """
    win = int(win_s * sr)
    hop = int(hop_s * sr)

    if win <= 0 or hop <= 0:
        raise ValueError("win_s and hop_s must be > 0")

    windows = []
    if n_samples < win:
        windows.append((0, win, 0.0, win_s))
        return windows

    for start in range(0, n_samples - win + 1, hop):
        end = start + win
        windows.append((start, end, start / sr, end / sr))

    # If audio is long but last tail is shorter than win, we ignore it for stability.
    return windows


def pad_to_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        return y
    return np.pad(y, (0, target_len - len(y)), mode="constant")


def percentile_aggregate(scores: List[float], q: float = 90.0) -> float:
    if not scores:
        return 0.0
    return float(np.percentile(np.asarray(scores, dtype=np.float32), q))


def find_drone_label_from_first_prediction(preds: List[Dict[str, Any]]) -> Optional[str]:
    """
    Try to identify the 'drone' label from model outputs.
    We prefer exact match, then substring match.
    """
    labels = [p["label"] for p in preds if "label" in p]
    # exact
    for l in labels:
        if l.strip().lower() == "drone":
            return l
    # contains
    for l in labels:
        if "drone" in l.lower():
            return l
    return None


def score_for_label(preds: List[Dict[str, Any]], target_label: str) -> float:
    for p in preds:
        if p.get("label") == target_label:
            return float(p.get("score", 0.0))
    return 0.0


def main():
    ap = argparse.ArgumentParser(
        description="Drone audio detection (file-based) using Hugging Face Transformers (CPU, no ONNX)."
    )
    ap.add_argument("audio_path", help="Path to audio file (wav/mp3/flac/etc.)")
    ap.add_argument("--model", default="preszzz/drone-audio-detection-05-17-trial-0",
                    help="HF model id or local path")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    ap.add_argument("--win", type=float, default=5.0, help="Window size in seconds")
    ap.add_argument("--hop", type=float, default=1.0, help="Hop size in seconds")
    ap.add_argument("--threshold", type=float, default=0.7, help="File-level decision threshold")
    ap.add_argument("--topk", type=int, default=5, help="How many hottest windows to report")
    ap.add_argument("--out", default="result.json", help="Output JSON file")
    ap.add_argument("--device", type=int, default=-1, help="HF pipeline device: -1 CPU, 0 GPU (not on Pi)")
    args = ap.parse_args()

    # 1) Load audio
    y, sr = load_audio_mono(args.audio_path, args.sr)

    win_len = int(args.win * sr)
    y = pad_to_length(y, win_len)

    # 2) Build pipeline (no training)
    # top_k=None -> returns all class scores (helps us find the 'drone' label reliably)
    clf = pipeline(
        "audio-classification",
        model=args.model,
        top_k=None,
        device=args.device
    )

    # 3) Create windows
    win_specs = sliding_windows(len(y), sr, args.win, args.hop)

    # 4) Run first window once to figure out label naming
    s0, e0, t0, t1 = win_specs[0]
    preds0 = clf({"array": y[s0:e0], "sampling_rate": sr})
    drone_label = find_drone_label_from_first_prediction(preds0)
    if drone_label is None:
        # Fail loudly with useful info
        labels = [p.get("label") for p in preds0]
        raise RuntimeError(
            "Could not find a 'drone' label in model outputs.\n"
            f"First-window labels returned by the model: {labels}\n"
            "If one of these corresponds to drone, set it manually in code."
        )

    results: List[WindowResult] = []
    scores: List[float] = []

    # 5) Score all windows
    for s, e, t_start, t_end in win_specs:
        chunk = y[s:e]
        preds = clf({"array": chunk, "sampling_rate": sr})
        score = score_for_label(preds, drone_label)

        results.append(WindowResult(t_start=t_start, t_end=t_end, score=score, label=drone_label))
        scores.append(score)

    # 6) Aggregate to file-level score
    file_score_p90 = percentile_aggregate(scores, q=90.0)
    decision = "drone" if file_score_p90 >= args.threshold else "no_drone"

    # 7) Pick hottest windows
    hottest = sorted(results, key=lambda r: r.score, reverse=True)[: args.topk]

    out: Dict[str, Any] = {
        "audio_path": args.audio_path,
        "model": args.model,
        "sample_rate": sr,
        "window_seconds": args.win,
        "hop_seconds": args.hop,
        "drone_label": drone_label,
        "file_score_p90": file_score_p90,
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
            "file_score_p90": file_score_p90,
            "drone_label": drone_label,
            "top_windows": out["top_windows"],
            "saved_to": args.out
        },
        indent=2
    ))


if __name__ == "__main__":
    main()