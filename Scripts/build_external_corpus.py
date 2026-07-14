#!/usr/bin/env python3
"""
build_external_corpus.py

Download publicly available ASR benchmark datasets via HuggingFace datasets,
convert audio to 16kHz mono WAV, and emit a corpus.jsonl file for the
voice-engine benchmark harness.

Datasets targeted (hard ASR, not LibriSpeech):
  - VoxPopuli (en, en_accented)     — European Parliament, political/technical
  - FLEURS (en_us)                  — multi-accent English
  - AMI (ihm, sdm)                  — meeting speech, close/distant mics
  - MInDS-14 (en-AU)                — spoken commands/payments, Australian English
  - People's Speech (dirty)         — noisy real-world speech

Each dataset contributes 80 clips to keep total disk usage moderate.

Usage:
    python3 Scripts/build_external_corpus.py

Output:
    bench_data/external_audio/     — 16kHz mono WAV files
    bench_data/corpus_external.jsonl — corpus entries for the bench harness
"""

import json
import os
import sys
import subprocess
import tempfile
import traceback
from pathlib import Path
from collections import Counter

import numpy as np
import soundfile as sf
from datasets import load_dataset, get_dataset_split_names
from datasets.features import Audio


# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "bench_data" / "external_audio"
CORPUS_PATH = PROJECT_ROOT / "bench_data" / "corpus_external.jsonl"
TARGET_SR = 16000
CLIPS_PER_DATASET = 80  # target; some datasets may have fewer in test split


# ── Dataset definitions ───────────────────────────────────────────────────────

DATASETS = [
    {
        "hf_id": "facebook/voxpopuli",
        "config": "en",
        "split": "test",
        "category": "voxpopuli-en",
        "text_field": "normalized_text",
        "max_clips": CLIPS_PER_DATASET,
    },
    {
        "hf_id": "facebook/voxpopuli",
        "config": "en_accented",
        "split": "test",
        "category": "voxpopuli-accented",
        "text_field": "normalized_text",
        "max_clips": CLIPS_PER_DATASET,
    },
    {
        "hf_id": "google/fleurs",
        "config": "en_us",
        "split": "test",
        "category": "fleurs-en-us",
        "text_field": "transcription",
        "max_clips": CLIPS_PER_DATASET,
    },
    {
        "hf_id": "edinburghcstr/ami",
        "config": "ihm",
        "split": "test",
        "category": "ami-ihm",
        "text_field": "text",
        "max_clips": CLIPS_PER_DATASET,
    },
    {
        "hf_id": "edinburghcstr/ami",
        "config": "sdm",
        "split": "test",
        "category": "ami-sdm",
        "text_field": "text",
        "max_clips": CLIPS_PER_DATASET,
    },
    {
        "hf_id": "PolyAI/minds14",
        "config": "en-AU",
        "split": "train",          # MInDS-14 only has 'train' split
        "category": "minds14-en-au",
        "text_field": "transcription",
        "max_clips": 80,
    },
    {
        "hf_id": "MLCommons/peoples_speech",
        "config": "dirty",
        "split": "test",
        "category": "peoples-speech-dirty",
        "text_field": "text",
        "max_clips": CLIPS_PER_DATASET,
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_and_resample(raw_bytes: bytes, out_path: Path) -> float:
    """Decode raw audio bytes and resample to 16kHz mono WAV via ffmpeg.
    Returns audio duration in seconds.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(raw_bytes)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_path,
                "-ar", str(TARGET_SR), "-ac", "1",
                "-sample_fmt", "s16",
                str(out_path),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        # Read the result to get duration
        data, _sr = sf.read(str(out_path))
        duration = len(data) / TARGET_SR
        return duration
    finally:
        os.unlink(tmp_path)


def load_dataset_no_torchcodec(hf_id: str, config: str, split: str):
    """Load a dataset split with audio decoding disabled to avoid torchcodec.

    torchcodec requires FFmpeg 4-7 at specific rpaths; this machine has
    FFmpeg 8 from Homebrew. Instead, we load raw bytes and decode ourselves.

    Returns an iterable where each row has audio 'bytes' and 'path' instead
    of the decoded 'array'.
    """
    ds = load_dataset(
        hf_id, config,
        split=split,
        streaming=True,
        trust_remote_code=False,
    )
    return ds.cast_column("audio", Audio(decode=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    all_entries = []
    stats = Counter()
    skipped = []

    for ds_cfg in DATASETS:
        hf_id = ds_cfg["hf_id"]
        config = ds_cfg["config"]
        category = ds_cfg["category"]
        text_field = ds_cfg["text_field"]
        max_clips = ds_cfg["max_clips"]

        print(f"\n{'='*60}")
        print(f"Dataset: {hf_id}  config={config}  --> category={category}")
        print(f"{'='*60}")

        # Validate dataset exists
        try:
            splits = get_dataset_split_names(hf_id, config)
        except Exception as e:
            msg = f"SKIP: cannot get splits -- {type(e).__name__}: {e}"
            print(f"  {msg}")
            skipped.append((f"{hf_id}/{config}", msg))
            continue

        split = ds_cfg["split"]
        if split not in splits:
            msg = f"SKIP: split '{split}' not found in {splits}"
            print(f"  {msg}")
            skipped.append((f"{hf_id}/{config}", msg))
            continue

        print(f"  Available splits: {splits}")

        # Load dataset (decode=False to avoid torchcodec issues)
        try:
            ds = load_dataset_no_torchcodec(hf_id, config, split)
        except Exception as e:
            msg = f"SKIP: cannot load -- {type(e).__name__}: {e}"
            print(f"  {msg}")
            skipped.append((f"{hf_id}/{config}", msg))
            continue

        downloaded = 0
        for idx, row in enumerate(ds):
            if downloaded >= max_clips:
                break

            # Extract text
            if text_field not in row:
                print(f"  WARN: row {idx} missing field '{text_field}', keys: {list(row.keys())[:8]}")
                continue
            ref_text = str(row[text_field]).strip()
            if not ref_text:
                continue

            # Extract raw audio bytes
            audio = row.get("audio")
            if audio is None:
                continue

            raw_bytes = audio.get("bytes")
            if raw_bytes is None:
                # Try to load from path if bytes aren't inlined
                continue

            # Skip if too small (likely corrupt or silence)
            if len(raw_bytes) < 1000:
                continue

            # Generate unique ID and output path
            safe_cat = category.replace("/", "-")
            uid = f"{safe_cat}_{downloaded:04d}"
            wav_path = AUDIO_DIR / f"{safe_cat}_{downloaded:04d}.wav"

            # Decode, resample, write final WAV
            try:
                duration = decode_and_resample(raw_bytes, wav_path)
            except Exception as e:
                print(f"  WARN: audio decode failed for row {idx}: {e}")
                continue

            # Skip very short or very long clips
            if duration < 0.5 or duration > 60:
                wav_path.unlink(missing_ok=True)
                continue

            entry = {
                "id": uid,
                "wav_path": str(wav_path),
                "ref_text": ref_text,
                "category": category,
                "audio_secs": round(duration, 2),
            }
            all_entries.append(entry)
            downloaded += 1

            if downloaded % 20 == 0:
                print(f"  ... {downloaded}/{max_clips} clips downloaded")

        stats[category] = downloaded
        print(f"  Done: {downloaded} clips saved (target was {max_clips})")

    # Write corpus file
    print(f"\n{'='*60}")
    print(f"Writing corpus --> {CORPUS_PATH}")
    print(f"{'='*60}")

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = sum(stats.values())
    print(f"Total clips: {total}")
    audio_bytes = sum(
        f.stat().st_size for f in AUDIO_DIR.glob("*.wav") if f.is_file()
    )
    print(f"Total audio size: {audio_bytes / (1024*1024):.1f} MB")
    print(f"Output: {CORPUS_PATH}")
    print()

    for category, count in sorted(stats.items()):
        print(f"  {category:30s}  {count:4d} clips")

    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if total == 0:
        print("\nERROR: No clips were successfully downloaded.")
        sys.exit(1)

    print(f"\nBenchmark command:")
    print(f"  swift run voice-engine --bench {CORPUS_PATH} --output-dir bench_data/results")
    print()


if __name__ == "__main__":
    main()
