#!/usr/bin/env python3
import argparse
import json
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import requests
import soundfile as sf
import librosa

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


def rms_normalize(y: np.ndarray, target_rms: float = 0.08, eps: float = 1e-8) -> np.ndarray:
    rms = np.sqrt(np.mean(y * y) + eps)
    gain = target_rms / max(rms, eps)
    y = y * gain
    return np.clip(y, -1.0, 1.0).astype(np.float32)

def save_wav(path: str, y: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr)


def post_detection(server_url: str, metadata: dict, audio_path: str):
    with open(audio_path, "rb") as f:
        files = {
            "audio": (os.path.basename(audio_path), f, "audio/wav")
        }
        data = {
            "metadata": json.dumps(metadata)
        }
        resp = requests.post(server_url, data=data, files=files, timeout=10)
        resp.raise_for_status()
        return resp.json()


def highpass_fft(y: np.ndarray, sr: int, cutoff_hz: float = 150.0) -> np.ndarray:
    """
    Simple FFT-domain high-pass filter.
    Good enough for MVP to remove low-frequency rumble.
    """
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    Y[freqs < cutoff_hz] = 0
    out = np.fft.irfft(Y, n=len(y))
    return out.astype(np.float32)


class SpectralNoiseReducer:
    """
    Keeps a running estimate of background magnitude spectrum and subtracts it.
    """
    def __init__(self, sr: int, n_fft: int = 1024, hop_length: int = 256, alpha: float = 1.5, floor: float = 0.05):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = alpha
        self.floor = floor
        self.noise_mag = None

    def update_noise_profile(self, y: np.ndarray):
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(D)
        current = mag.mean(axis=1, keepdims=True)

        if self.noise_mag is None:
            self.noise_mag = current
        else:
            # Slow EMA so profile adapts gently
            self.noise_mag = 0.95 * self.noise_mag + 0.05 * current

    def reduce(self, y: np.ndarray) -> np.ndarray:
        if self.noise_mag is None:
            return y.astype(np.float32)

        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        mag = np.abs(D)
        phase = np.angle(D)

        clean_mag = mag - self.alpha * self.noise_mag
        clean_mag = np.maximum(clean_mag, self.floor * mag)

        D_clean = clean_mag * np.exp(1j * phase)
        y_clean = librosa.istft(D_clean, hop_length=self.hop_length, length=len(y))
        return np.clip(y_clean, -1.0, 1.0).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-fifo", required=True)
    ap.add_argument("--onnx_dir", default="onnx_drone_model")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--history", type=int, default=5)
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--out", default="")

    # Preprocessing params
    ap.add_argument("--hp-cutoff", type=float, default=150.0, help="High-pass cutoff frequency in Hz")
    ap.add_argument("--target-rms", type=float, default=0.08, help="Target RMS after normalization")
    ap.add_argument("--noise-alpha", type=float, default=1.5, help="Spectral subtraction strength")
    ap.add_argument("--noise-floor", type=float, default=0.05, help="Residual floor to avoid artifacts")
    ap.add_argument("--strict-bg-update", action="store_true", help="Update noise profile only when score is very low")
    ap.add_argument("--clip-dir", default="detected_clips")

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

    noise_reducer = SpectralNoiseReducer(
        sr=args.sr,
        n_fft=1024,
        hop_length=256,
        alpha=args.noise_alpha,
        floor=args.noise_floor,
    )

    win_samples = int(args.win * args.sr)
    hop_samples = int(args.hop * args.sr)
    chunk_samples = int(args.sr * (args.chunk_ms / 1000.0))
    chunk_bytes = chunk_samples * 4  # float32

    ring_buffer = np.zeros(win_samples, dtype=np.float32)
    filled = 0
    since_last_infer = 0
    stream_time_s = 0.0

    recent_scores = deque(maxlen=args.history)
    prev_decision = None
    prev_raw_score = 0.0

    gps_lat, gps_lon = 44.436142, 26.102684  # Placeholder GPS coordinates
    server_url = "http://2doorspacemachine.local:8001/events"  # Placeholder server URL
    out_file = open(args.out, "a", encoding="utf-8") if args.out else None

    print(f"Listening on FIFO: {args.input_fifo}")
    print(f"Using label: {drone_label}")

    try:
        with open(args.input_fifo, "rb") as f:
            while True:
                raw = f.read(chunk_bytes)
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

                window = ring_buffer.copy().astype(np.float32)
                window = window - np.mean(window)
                window = rms_normalize(window, target_rms=0.08)

                since_last_infer = 0

                raw_score = score_chunk(clf, window, args.sr, drone_label)
                prev_raw_score = raw_score

                recent_scores.append(raw_score)

                # More responsive than percentile for short mixed events
                smoothed_score = max(recent_scores) if recent_scores else raw_score

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
                # Only post on transition into drone state
                if decision == "drone" and prev_decision != "drone":
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    clip_path = os.path.join(args.clip_dir, f"drone_detected_{timestamp}.wav")
                    save_wav(clip_path, window, args.sr)

                    metadata = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "gps": {
                            "lat": gps_lat,
                            "lon": gps_lon,
                        },
                        "event": "drone detected",
                        "type": "fpv",
                        "confidence": smoothed_score,
                        "window": {
                            "t_start": event["t_start"],
                            "t_end": event["t_end"],
                        },
                    }

                    try:
                        response_json = post_detection(server_url, metadata, clip_path)
                        print(f"POST OK: {response_json}")
                    except Exception as e:
                        print(f"POST FAILED: {e}")

                prev_decision = decision

    finally:
        if out_file:
            out_file.close()


if __name__ == "__main__":
    main()