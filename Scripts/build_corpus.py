#!/usr/bin/env python3
"""
Build bench_data/corpus.jsonl from the audio directory.

Pairs WAV files with their transcription JSON sidecars (90 pairs).
Classifies utterances into categories using simple heuristics.
WAV-only files (no JSON) get empty ref_text — they need manual transcription.
"""
import json
import os
import re
import sys
from pathlib import Path

AUDIO_DIR = Path.home() / "Library/Logs/voice-engine/audio"
OUTPUT_PATH = "bench_data/corpus.jsonl"

# Heuristic category classification
QUESTION_PATTERNS = [
    r"^what\b", r"^why\b", r"^how\b", r"^when\b", r"^where\b", r"^who\b",
    r"^can\b", r"^could\b", r"^would\b", r"^should\b", r"^will\b",
    r"^is\b", r"^are\b", r"^was\b", r"^were\b", r"^do\b", r"^does\b",
    r"^did\b", r"^has\b", r"^have\b", r"^am\b",
]
SELF_CORRECTION_MARKERS = [
    "actually", "i mean", "sorry", "no wait", "let me rephrase",
    "that's not", "scratch that", "never mind",
]


def classify(text: str) -> str:
    """Heuristic category classifier."""
    if not text.strip():
        return "unknown"
    t = text.strip().lower()

    # Question: ends with ? or starts with question word
    if t.endswith("?"):
        return "question"
    for pat in QUESTION_PATTERNS:
        if re.search(pat, t):
            return "question"

    # Self-correction
    for marker in SELF_CORRECTION_MARKERS:
        if marker in t:
            return "self_correction"

    # Code/commands: contains programming keywords or shell patterns
    code_markers = [
        "def ", "class ", "import ", "from ", "return ", "func ", "const ",
        "let ", "var ", "sudo ", "pip ", "npm ", "git ", "docker ",
        "ssh ", "curl ", "grep ", "|", "&&", "./", "=>",
    ]
    for marker in code_markers:
        if marker in t:
            return "code_command"

    # Technical: contains tech vocabulary
    tech_terms = [
        "api", "json", "http", "server", "client", "database", "token",
        "encoder", "decoder", "model", "inference", "latency", "pipeline",
        "coreml", "ane", "gpu", "cpu", "memory", "cache", "kernel",
        "transcription", "wer", "benchmark",
    ]
    for term in tech_terms:
        if term in t:
            return "technical"

    return "dictation"


def load_json_robust(path: Path) -> dict:
    """Load a JSON file, handling malformed content (extra braces, trailing data)."""
    content = path.read_text().strip()
    # Try standard parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Try finding the first complete JSON object
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(content)
        return obj
    except json.JSONDecodeError:
        pass
    # Last resort: strip trailing braces/whitespace
    # Common issue: double }} at end
    while content.endswith("}"):
        content = content.rstrip().rstrip("}").rstrip()
        try:
            return json.loads(content + "}")
        except json.JSONDecodeError:
            continue
    print(f"WARNING: Could not parse {path}", file=sys.stderr)
    return {}


def main():
    if not AUDIO_DIR.exists():
        print(f"Audio directory not found: {AUDIO_DIR}", file=sys.stderr)
        sys.exit(1)

    wav_files = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
    json_files = {f.replace(".json", ""): f for f in os.listdir(AUDIO_DIR) if f.endswith(".json")}

    print(f"WAV files: {len(wav_files)}")
    print(f"JSON sidecars: {len(json_files)}")

    entries = []
    paired = 0
    unpaired = 0

    for wav_name in wav_files:
        ts = wav_name.replace(".wav", "")
        wav_path = str(AUDIO_DIR / wav_name)

        if ts in json_files:
            # Read JSON sidecar for reference text and metadata
            json_path = AUDIO_DIR / json_files[ts]
            data = load_json_robust(json_path)
            ref_text = data.get("text", "").strip()
            audio_secs = data.get("audio_secs")
            paired += 1
        else:
            # No transcription available — needs manual review
            ref_text = ""
            audio_secs = None
            unpaired += 1

        category = classify(ref_text) if ref_text else "unknown"

        entry = {
            "id": ts,
            "wav_path": wav_path,
            "ref_text": ref_text,
            "category": category,
            "audio_secs": audio_secs,
        }
        entries.append(entry)

    # Write corpus.jsonl
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Summary
    categories = {}
    for e in entries:
        cat = e["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nCorpus written to {OUTPUT_PATH}")
    print(f"  Paired (with transcript): {paired}")
    print(f"  Unpaired (needs manual):  {unpaired}")
    print(f"  Total: {len(entries)}")
    print(f"\nCategories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
