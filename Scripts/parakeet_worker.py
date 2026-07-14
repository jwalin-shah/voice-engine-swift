#!/usr/bin/env python3
"""
Persistent MLX Parakeet worker: loads the model once, listens on stdin for
audio file paths, returns transcriptions on stdout as JSON.

Input (stdin, one JSON object per line):
  {"path": "/path/to/audio.wav"}

Output (stdout, one JSON object per line):
  {"text": "transcribed text", "total_ms": 42.5}

Exits cleanly on EOF or a line starting with "EXIT".
"""

import json
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
from parakeet_mlx import from_pretrained

MODEL_PATH = "/tmp/parakeet-mlx-cache/models--senstella--parakeet-tdt-0.6b-v2-mlx/snapshots/ab487198eff120b8e606184c7fc733fd605e7fc0"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main():
    log("[worker] loading model...")
    t0 = time.perf_counter()
    model = from_pretrained(MODEL_PATH)
    log(f"[worker] model loaded in {(time.perf_counter() - t0) * 1000:.0f}ms")

    # Warm up: the first real call triggers MLX compilation (JIT),
    # so we accept a warmup request or just signal ready
    log("[worker] ready (warmup will happen on first real call)")

    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("EXIT"):
            break

        try:
            req = json.loads(line)
            path = req["path"]

            t0 = time.perf_counter()
            result = model.transcribe(path)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            resp = {"text": result.text, "total_ms": round(elapsed_ms, 3)}
            print(json.dumps(resp), flush=True)

        except Exception:
            tb = traceback.format_exc()
            log(f"[worker] error: {tb}")
            resp = {"text": "", "total_ms": 0, "error": tb[:200]}
            print(json.dumps(resp), flush=True)

    log("[worker] exiting")


if __name__ == "__main__":
    main()
