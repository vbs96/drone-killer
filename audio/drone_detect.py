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

import sounddevice as sd

import queue
import threading
import time


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

def audio_chunks_from_fifo(input_fifo: str, chunk_bytes: int):
    with open(input_fifo, "rb") as f:
        while True:
            raw = f.read(chunk_bytes)
            if not raw or len(raw) < chunk_bytes:
                continue
            yield np.frombuffer(raw, dtype=np.float32)

def audio_chunks_from_mic(sr: int, chunk_samples: int, device=None, channels: int = 1):
    with sd.InputStream(
        samplerate=sr,
        channels=channels,
        dtype="float32",
        blocksize=chunk_samples,
        device=device,
    ) as stream:
        while True:
            data, overflowed = stream.read(chunk_samples)
            if overflowed:
                print("Warning: microphone input overflowed")
            # data shape: (frames, channels)
            if channels > 1:
                chunk = np.mean(data, axis=1, dtype=np.float32)
            else:
                chunk = data[:, 0].astype(np.float32)
            yield chunk

def start_mic_stream(audio_queue: "queue.Queue[np.ndarray]", sr: int, chunk_samples: int, device=None, channels: int = 1):
    """
    Starts a sounddevice InputStream whose callback only copies audio
    into a queue as fast as possible.

    Returns the opened stream object. Caller must keep it alive and close it.
    """

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Mic status: {status}")

        try:
            # Convert to mono float32 with minimal work
            if channels > 1:
                chunk = np.mean(indata, axis=1, dtype=np.float32)
            else:
                chunk = indata[:, 0].copy()

            # Non-blocking put: if full, drop oldest item to keep real-time behavior
            try:
                audio_queue.put_nowait(chunk)
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    audio_queue.put_nowait(chunk)
                except queue.Full:
                    pass
        except Exception as e:
            print(f"Audio callback error: {e}")

    stream = sd.InputStream(
        samplerate=sr,
        channels=channels,
        dtype="float32",
        blocksize=chunk_samples,
        latency="high",
        device=device,
        callback=audio_callback,
    )
    stream.start()
    return stream


def detector_worker(
    audio_queue: "queue.Queue[np.ndarray]",
    post_queue: "queue.Queue[dict]",
    stop_event: threading.Event,
    args,
    clf,
    drone_label: str,
    gps_lat: float,
    gps_lon: float,
    out_file,
):
    win_samples = int(args.win * args.sr)
    hop_samples = int(args.hop * args.sr)

    ring_buffer = np.zeros(win_samples, dtype=np.float32)
    filled = 0
    since_last_infer = 0
    stream_time_s = 0.0

    recent_scores = deque(maxlen=args.history)
    prev_decision = None
    prev_raw_score = 0.0

    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

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

        # Keep only the preprocessing you actually want active
        window = rms_normalize(window, target_rms=args.target_rms)

        since_last_infer = 0

        raw_score = score_chunk(clf, window, args.sr, drone_label)
        prev_raw_score = raw_score

        recent_scores.append(raw_score)
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

        # Only enqueue post/save when entering drone state
        if decision == "drone" and prev_decision != "drone":
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            clip_path = os.path.join(args.clip_dir, f"drone_detected_{timestamp}.wav")

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

            job = {
                "clip_path": clip_path,
                "audio": window.copy(),
                "sr": args.sr,
                "metadata": metadata,
            }

            try:
                post_queue.put_nowait(job)
            except queue.Full:
                print("Warning: post queue full, dropping detection upload job")

        prev_decision = decision


def post_worker(
    post_queue: "queue.Queue[dict]",
    stop_event: threading.Event,
    server_url: str,
):
    while not stop_event.is_set():
        try:
            job = post_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            clip_path = job["clip_path"]
            audio = job["audio"]
            sr = job["sr"]
            metadata = job["metadata"]

            save_wav(clip_path, audio, sr)
            response_json = post_detection(server_url, metadata, clip_path)
            print(f"POST OK: {response_json}")
        except Exception as e:
            print(f"POST FAILED: {e}")
        finally:
            post_queue.task_done()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-fifo", default=None)
    ap.add_argument("--mic", action="store_true", help="Read from microphone instead of FIFO")
    ap.add_argument("--device", default=None, help="Input device name or index")
    ap.add_argument("--channels", type=int, default=1, help="Number of input channels")

    ap.add_argument("--onnx_dir", default="onnx_drone_model")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--history", type=int, default=5)
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--out", default="")

    ap.add_argument("--hp-cutoff", type=float, default=150.0, help="High-pass cutoff frequency in Hz")
    ap.add_argument("--target-rms", type=float, default=0.08, help="Target RMS after normalization")
    ap.add_argument("--noise-alpha", type=float, default=1.5, help="Spectral subtraction strength")
    ap.add_argument("--noise-floor", type=float, default=0.05, help="Residual floor to avoid artifacts")
    ap.add_argument("--strict-bg-update", action="store_true", help="Update noise profile only when score is very low")
    ap.add_argument("--clip-dir", default="detected_clips")

    ap.add_argument("--audio-queue-size", type=int, default=64, help="Max queued mic chunks")
    ap.add_argument("--post-queue-size", type=int, default=16, help="Max queued post jobs")

    args = ap.parse_args()

    if not args.mic and not args.input_fifo:
        ap.error("Either --mic or --input-fifo must be provided")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.onnx_dir)
    ort_model = ORTModelForAudioClassification.from_pretrained(args.onnx_dir)

    clf = pipeline(
        task="audio-classification",
        model=ort_model,
        feature_extractor=feature_extractor,
        top_k=None,
    )

    drone_label = find_drone_label(ort_model)

    chunk_samples = int(args.sr * (args.chunk_ms / 1000.0))

    gps_lat, gps_lon = 44.436142, 26.102684
    server_url = "http://2doorspacemachine.local:8001/events"

    out_file = open(args.out, "a", encoding="utf-8") if args.out else None

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=args.audio_queue_size)
    post_queue: queue.Queue[dict] = queue.Queue(maxsize=args.post_queue_size)
    stop_event = threading.Event()

    detector_thread = None
    post_thread = None
    mic_stream = None

    print(f"Using label: {drone_label}")

    try:
        post_thread = threading.Thread(
            target=post_worker,
            args=(post_queue, stop_event, server_url),
            daemon=True,
        )
        post_thread.start()

        detector_thread = threading.Thread(
            target=detector_worker,
            args=(
                audio_queue,
                post_queue,
                stop_event,
                args,
                clf,
                drone_label,
                gps_lat,
                gps_lon,
                out_file,
            ),
            daemon=True,
        )
        detector_thread.start()

        if args.mic:
            print(f"Listening on microphone: device={args.device}, channels={args.channels}, sr={args.sr}")
            mic_stream = start_mic_stream(
                audio_queue=audio_queue,
                sr=args.sr,
                chunk_samples=chunk_samples,
                device=args.device,
                channels=args.channels,
            )

            while True:
                time.sleep(1.0)

        else:
            print(f"Listening on FIFO: {args.input_fifo}")
            chunk_bytes = chunk_samples * 4

            with open(args.input_fifo, "rb") as f:
                while True:
                    raw = f.read(chunk_bytes)
                    if not raw or len(raw) < chunk_bytes:
                        continue

                    chunk = np.frombuffer(raw, dtype=np.float32).copy()

                    try:
                        audio_queue.put(chunk, timeout=1.0)
                    except queue.Full:
                        print("Warning: audio queue full, dropping FIFO chunk")

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        stop_event.set()

        if mic_stream is not None:
            try:
                mic_stream.stop()
                mic_stream.close()
            except Exception:
                pass

        if detector_thread is not None:
            detector_thread.join(timeout=2.0)

        if post_thread is not None:
            post_thread.join(timeout=2.0)

        if out_file:
            out_file.close()


if __name__ == "__main__":
    main()