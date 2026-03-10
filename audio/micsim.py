#!/usr/bin/env python3
import argparse
import sys
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Iterator

import numpy as np
import librosa


@dataclass
class DroneEvent:
    start_t: float
    end_t: float
    fade_in_s: float
    fade_out_s: float
    peak_gain: float
    drone_offset_samples: int


def load_audio_mono(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")
    y = np.nan_to_num(y).astype(np.float32)
    return y, target_sr


def loop_read(buf: np.ndarray, idx: int, n: int) -> Tuple[np.ndarray, int]:
    out = np.empty(n, dtype=np.float32)
    remaining = n
    out_i = 0
    L = len(buf)

    while remaining > 0:
        if idx >= L:
            idx = 0
        take = min(remaining, L - idx)
        out[out_i:out_i + take] = buf[idx:idx + take]
        idx += take
        out_i += take
        remaining -= take

    return out, idx


class MultiBackgroundPlayer:
    def __init__(self, backgrounds: List[np.ndarray], rng: random.Random):
        if not backgrounds:
            raise ValueError("At least one background file is required.")
        self.backgrounds = backgrounds
        self.rng = rng
        self.current_idx = 0
        self.current_pos = 0

    def read(self, n: int) -> np.ndarray:
        bg = self.backgrounds[self.current_idx]
        chunk, self.current_pos = loop_read(bg, self.current_pos, n)

        # Occasionally switch to another background when looping back near start
        if self.current_pos < n and len(self.backgrounds) > 1:
            self.current_idx = self.rng.randrange(len(self.backgrounds))
            self.current_pos = 0

        return chunk


class SimMicV2:
    def __init__(
        self,
        backgrounds: List[np.ndarray],
        drone: np.ndarray,
        sr: int,
        chunk_ms: int,
        event_every_s: float,
        event_duration_min_s: float,
        event_duration_max_s: float,
        fade_min_s: float,
        fade_max_s: float,
        peak_gain_min: float,
        peak_gain_max: float,
        bg_gain: float,
        seed: Optional[int] = None,
    ):
        self.sr = sr
        self.chunk_size = int(sr * (chunk_ms / 1000.0))
        self.rng = random.Random(seed)

        self.bg_player = MultiBackgroundPlayer(backgrounds, self.rng)
        self.drone = drone
        self.drone_len = len(drone)

        self.event_every_s = event_every_s
        self.event_duration_min_s = event_duration_min_s
        self.event_duration_max_s = event_duration_max_s
        self.fade_min_s = fade_min_s
        self.fade_max_s = fade_max_s
        self.peak_gain_min = peak_gain_min
        self.peak_gain_max = peak_gain_max
        self.bg_gain = bg_gain

        self.stream_start_t = time.monotonic()
        self.next_event_t = 0.0
        self.active_event: Optional[DroneEvent] = None

        self._schedule_next_event(0.0)

    def _schedule_next_event(self, now_stream_t: float):
        jitter = self.rng.uniform(0.6, 1.4)
        self.next_event_t = now_stream_t + self.event_every_s * jitter

    def _create_event(self, now_stream_t: float) -> DroneEvent:
        duration_s = self.rng.uniform(self.event_duration_min_s, self.event_duration_max_s)
        fade_in_s = self.rng.uniform(self.fade_min_s, self.fade_max_s)
        fade_out_s = self.rng.uniform(self.fade_min_s, self.fade_max_s)

        max_fade_total = max(0.1, duration_s - 0.5)
        if fade_in_s + fade_out_s > max_fade_total:
            scale = max_fade_total / (fade_in_s + fade_out_s)
            fade_in_s *= scale
            fade_out_s *= scale

        peak_gain = self.rng.uniform(self.peak_gain_min, self.peak_gain_max)
        drone_offset_samples = self.rng.randrange(max(1, self.drone_len))

        return DroneEvent(
            start_t=now_stream_t,
            end_t=now_stream_t + duration_s,
            fade_in_s=fade_in_s,
            fade_out_s=fade_out_s,
            peak_gain=peak_gain,
            drone_offset_samples=drone_offset_samples,
        )

    def _event_gain_at(self, event: DroneEvent, t: float) -> float:
        if t < event.start_t or t >= event.end_t:
            return 0.0

        if t < event.start_t + event.fade_in_s:
            progress = (t - event.start_t) / max(event.fade_in_s, 1e-6)
            return event.peak_gain * progress

        if t > event.end_t - event.fade_out_s:
            progress = (event.end_t - t) / max(event.fade_out_s, 1e-6)
            return event.peak_gain * progress

        return event.peak_gain

    def _drone_chunk_with_envelope(self, event: DroneEvent, chunk_stream_start_t: float, n: int) -> np.ndarray:
        drone_chunk, event.drone_offset_samples = loop_read(self.drone, event.drone_offset_samples, n)

        # Per-sample envelope for smooth approach/departure
        sample_times = chunk_stream_start_t + (np.arange(n, dtype=np.float32) / self.sr)
        gains = np.array([self._event_gain_at(event, float(t)) for t in sample_times], dtype=np.float32)

        return drone_chunk * gains

    def stream(self) -> Iterator[Tuple[np.ndarray, dict]]:
        chunk_seconds = self.chunk_size / self.sr

        while True:
            now_stream_t = time.monotonic() - self.stream_start_t

            # Start new event if needed
            event_started = False
            if self.active_event is None and now_stream_t >= self.next_event_t:
                self.active_event = self._create_event(now_stream_t)
                self._schedule_next_event(now_stream_t)
                event_started = True

            bg_chunk = self.bg_player.read(self.chunk_size) * self.bg_gain

            meta = {
                "stream_t": now_stream_t,
                "event_active": False,
                "event_started": event_started,
                "event_ended": False,
                "event_peak_gain": 0.0,
                "event_gain_now": 0.0,
                "event_start_t": None,
                "event_end_t": None,
            }

            if self.active_event is not None:
                event = self.active_event
                drone_chunk = self._drone_chunk_with_envelope(event, now_stream_t, self.chunk_size)
                gain_now = self._event_gain_at(event, now_stream_t)

                # During an event, drone audio replaces background instead of overlaying it.
                out = drone_chunk
                meta.update({
                    "event_active": True,
                    "event_peak_gain": event.peak_gain,
                    "event_gain_now": gain_now,
                    "event_start_t": event.start_t,
                    "event_end_t": event.end_t,
                })

                if now_stream_t >= event.end_t:
                    meta["event_ended"] = True
                    self.active_event = None
            else:
                out = bg_chunk

            out = np.clip(out, -1.0, 1.0).astype(np.float32)

            yield out, meta
            time.sleep(chunk_seconds)


def parse_args():
    ap = argparse.ArgumentParser(description="Simulated microphone v2: realistic background + drone events.")
    ap.add_argument("--background", required=True, nargs="+", help="One or more background audio files")
    ap.add_argument("--drone", required=True, help="Drone audio file")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--chunk-ms", type=int, default=100)

    ap.add_argument("--event-every", type=float, default=20.0, help="Average seconds between drone events")
    ap.add_argument("--event-duration-min", type=float, default=8.0)
    ap.add_argument("--event-duration-max", type=float, default=14.0)

    ap.add_argument("--fade-min", type=float, default=1.5)
    ap.add_argument("--fade-max", type=float, default=3.0)

    ap.add_argument("--peak-gain-min", type=float, default=0.20, help="Lower = farther / harder")
    ap.add_argument("--peak-gain-max", type=float, default=0.75, help="Higher = closer / easier")

    ap.add_argument("--bg-gain", type=float, default=0.9, help="Background gain multiplier")
    ap.add_argument("--seed", type=int, default=0)

    return ap.parse_args()


def main():
    args = parse_args()

    backgrounds = []
    for bg_path in args.background:
        y, _ = load_audio_mono(bg_path, args.sr)
        backgrounds.append(y)

    drone, sr = load_audio_mono(args.drone, args.sr)

    sim = SimMicV2(
        backgrounds=backgrounds,
        drone=drone,
        sr=sr,
        chunk_ms=args.chunk_ms,
        event_every_s=args.event_every,
        event_duration_min_s=args.event_duration_min,
        event_duration_max_s=args.event_duration_max,
        fade_min_s=args.fade_min,
        fade_max_s=args.fade_max,
        peak_gain_min=args.peak_gain_min,
        peak_gain_max=args.peak_gain_max,
        bg_gain=args.bg_gain,
        seed=args.seed,
    )

    try:
        for chunk, meta in sim.stream():
            # stdout = raw float32 PCM for the detector
            sys.stdout.buffer.write(chunk.tobytes())
            sys.stdout.buffer.flush()

            # stderr = readable ground truth for you
            if meta["event_started"]:
                print(
                    f"[EVENT_START] t={meta['stream_t']:.2f}s "
                    f"start={meta['event_start_t']:.2f}s end={meta['event_end_t']:.2f}s "
                    f"peak_gain={meta['event_peak_gain']:.2f}",
                    file=sys.stderr,
                    flush=True,
                )

            if meta["event_active"]:
                print(
                    f"[EVENT_ACTIVE] t={meta['stream_t']:.2f}s "
                    f"gain_now={meta['event_gain_now']:.2f} peak_gain={meta['event_peak_gain']:.2f}",
                    file=sys.stderr,
                    flush=True,
                )

            if meta["event_ended"]:
                print(
                    f"[EVENT_END] t={meta['stream_t']:.2f}s",
                    file=sys.stderr,
                    flush=True,
                )

    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()