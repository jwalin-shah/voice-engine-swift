#!/usr/bin/env python3
"""
Transcribe WAV files using nvidia/parakeet-tdt-0.6b-v2 (NeMo).

Usage:
    python3 Scripts/parakeet_transcribe.py <wav_path>
    python3 Scripts/parakeet_transcribe.py <wav_path_1> <wav_path_2> ...

Outputs one JSON object per file to stdout (jsonl).
"""
import json
import sys
import time
from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch


MODEL_PATH = "/tmp/parakeet/parakeet-tdt-0.6b-v2.nemo"


def load_model(device: str = "mps"):
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(MODEL_PATH)
    model = model.eval()
    model = model.to(device)
    return model


def transcribe_file(model, wav_path: str) -> dict:
    start = time.perf_counter()
    results = model.transcribe([str(wav_path)], batch_size=1)
    elapsed_ms = (time.perf_counter() - start) * 1000

    hyp = results[0]
    return {
        "wav_path": str(wav_path),
        "hyp_text": hyp.text,
        "total_ms": round(elapsed_ms, 3),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: parakeet_transcribe.py <wav_path> [wav_path ...]", file=sys.stderr)
        sys.exit(1)

    wav_paths = sys.argv[1:]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading model on {device}...", file=sys.stderr)
    model = load_model(device)
    print(f"Model loaded.", file=sys.stderr)

    # Warmup
    _ = model.transcribe([str(wav_paths[0])], batch_size=1)

    for wav_path in wav_paths:
        result = transcribe_file(model, wav_path)
        print(json.dumps(result))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
