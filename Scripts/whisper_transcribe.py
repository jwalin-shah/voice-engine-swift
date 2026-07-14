#!/usr/bin/env python3
"""
whisper_transcribe.py - Run whisper.cpp against a single WAV file.

Usage:
    python3 whisper_transcribe.py <wav_path> [--model small.en] [--model-path /path/to/model.bin]

Outputs JSON to stdout with transcribed text and timing breakdown.
"""
import json, re, subprocess, sys, tempfile, os, argparse
from pathlib import Path

WHISPER_CLI = "/tmp/whisper.cpp/build/bin/whisper-cli"
MODEL_DIR = "/tmp/whisper.cpp/models"


def transcribe(wav_path: str, model_name: str = "small.en", model_path: str = None) -> dict:
    """Run whisper-cli on a WAV file, return dict with text + timing."""
    if model_path is None:
        model_path = f"{MODEL_DIR}/ggml-{model_name}.bin"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(wav_path).exists():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            WHISPER_CLI,
            "-m", model_path,
            "-f", wav_path,
            "--no-timestamps",
            "-oj",
            "-of", tmp_path.replace(".json", ""),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stderr = result.stderr

        # Parse JSON output for transcribed text
        json_path = tmp_path.replace(".json", "") + ".json"
        if not Path(json_path).exists():
            # whisper-cli appends .json itself
            json_path = tmp_path

        text = ""
        if Path(json_path).exists():
            try:
                data = json.loads(Path(json_path).read_text())
                segments = data.get("transcription", [])
                if segments:
                    text = segments[0].get("text", "").strip()
            except (json.JSONDecodeError, KeyError):
                pass

        # Parse timing from stderr
        timing = parse_timing(stderr)

        return {"text": text, "timing": timing, "stderr": stderr}

    finally:
        for p in [tmp_path, tmp_path.replace(".json", "") + ".json", tmp_path.replace(".json", "")]:
            if Path(p).exists():
                try:
                    os.unlink(p)
                except OSError:
                    pass


def parse_timing(stderr: str) -> dict:
    """Parse whisper_print_timings lines from stderr."""
    timing = {}
    patterns = {
        "load_ms": r"load time\s*=\s*([\d.]+)\s*ms",
        "fallbacks_p": r"fallbacks\s*=\s*(\d+)\s*p",
        "fallbacks_h": r"fallbacks\s*=\s*\d+\s*p\s*/\s*(\d+)\s*h",
        "mel_ms": r"mel time\s*=\s*([\d.]+)\s*ms",
        "sample_ms": r"sample time\s*=\s*([\d.]+)\s*ms",
        "encode_ms": r"encode time\s*=\s*([\d.]+)\s*ms",
        "decode_ms": r"decode time\s*=\s*([\d.]+)\s*ms",
        "batchd_ms": r"batchd time\s*=\s*([\d.]+)\s*ms",
        "prompt_ms": r"prompt time\s*=\s*([\d.]+)\s*ms",
        "total_ms": r"total time\s*=\s*([\d.]+)\s*ms",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, stderr)
        if m:
            timing[key] = float(m.group(1))

    # Compute processing_ms = total_ms - load_ms (exclude model loading)
    if "total_ms" in timing:
        timing["processing_ms"] = timing["total_ms"] - timing.get("load_ms", 0)

    return timing


def main():
    parser = argparse.ArgumentParser(description="Transcribe a WAV file with whisper.cpp")
    parser.add_argument("wav_path", help="Path to WAV file")
    parser.add_argument("--model", default="small.en", help="Model name (default: small.en)")
    parser.add_argument("--model-path", default=None, help="Full path to model file")
    args = parser.parse_args()

    result = transcribe(args.wav_path, args.model, args.model_path)

    # Print clean output
    output = {
        "text": result["text"],
        "timing": result["timing"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
