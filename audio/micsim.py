#!/usr/bin/env python3
import argparse
import sys
import time
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Iterator

import numpy as np
import librosa


@dataclass
class InjectPlan:
    every_s: float
    duration_s: float
    mode: str
    gain: float
    bg_gain: float


def load_audio_mono(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")
    y = np.nan_to_num(y).astype(np.float32)
    return y, target_sr


class SimMic:
    def __init__(
        self,
        background: np.ndarray,
        drone: np.ndarray,
        sr: int,
        chunk_ms: int,
        plan: InjectPlan,
        seed: Optional[int] = None,
    ):
        self.bg = background
        self.drone = drone
        self.sr = sr
        self.chunk_size = int(sr * (chunk_ms / 1000.0))
        self.plan = plan
        self.rng = random.Random(seed)

        self.bg_i = 0
        self.drone_i = 0
        self.inject_until_t = 0.0
        self.next_inject_t = 0.0
        self._schedule_next_inject(time.monotonic())

    def _schedule_next_inject(self, now_t: float):
        jitter = self.rng.uniform(0.5, 1.5)
        self.next_inject_t = now_t + self.plan.every_s * jitter

    def _loop_read(self, buf: np.ndarray, idx: int, n: int):
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

    def stream(self) -> Iterator[Tuple[np.ndarray, bool]]:
        chunk_seconds = self.chunk_size / self.sr

        while True:
            now = time.monotonic()

            if now >= self.next_inject_t and now >= self.inject_until_t:
                self.inject_until_t = now + self.plan.duration_s
                self.drone_i = 0
                self._schedule_next_inject(now)

            injecting = now < self.inject_until_t

            bg_chunk, self.bg_i = self._loop_read(self.bg, self.bg_i, self.chunk_size)

            if not injecting:
                out = bg_chunk
            else:
                drone_chunk, self.drone_i = self._loop_read(self.drone, self.drone_i, self.chunk_size)
                if self.plan.mode == "replace":
                    out = self.plan.gain * drone_chunk
                else:
                    out = self.plan.bg_gain * bg_chunk + self.plan.gain * drone_chunk

            out = np.tanh(out).astype(np.float32)
            yield out, injecting
            time.sleep(chunk_seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", required=True)
    ap.add_argument("--drone", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--inject-every", type=float, default=20.0)
    ap.add_argument("--inject-duration", type=float, default=6.0)
    ap.add_argument("--inject-mode", choices=["overlay", "replace"], default="overlay")
    ap.add_argument("--inject-gain", type=float, default=1.0)
    ap.add_argument("--bg-gain", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bg, sr = load_audio_mono(args.background, args.sr)
    dr, _ = load_audio_mono(args.drone, args.sr)

    sim = SimMic(
        background=bg,
        drone=dr,
        sr=sr,
        chunk_ms=args.chunk_ms,
        plan=InjectPlan(
            every_s=args.inject_every,
            duration_s=args.inject_duration,
            mode=args.inject_mode,
            gain=args.inject_gain,
            bg_gain=args.bg_gain,
        ),
        seed=args.seed,
    )

    try:
        for chunk, injecting in sim.stream():
            # write raw float32 PCM to stdout
            sys.stdout.buffer.write(chunk.tobytes())
            sys.stdout.buffer.flush()

            # log to stderr so stdout stays pure audio
            state = "SIM-DRONE" if injecting else "BG"
            print(f"[{state}] wrote {len(chunk)} samples", file=sys.stderr)
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()