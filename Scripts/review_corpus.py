#!/usr/bin/env python3
"""
Review tool for ASR benchmark corpus.

Plays WAV files one at a time and shows the reference transcript (from JSON
sidecar) alongside what the current model transcribed. Lets you correct or
confirm each transcript, updating the JSON sidecar in place.

Requires: afplay (built-in macOS)

Usage:
    python3 Scripts/review_corpus.py [--start ID] [--limit N]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

AUDIO_DIR = Path.home() / "Library/Logs/voice-engine/audio"
RESULTS_FILE = Path("bench_data/results/moonshine-tiny.jsonl")


def load_results():
    """Load benchmark results, keyed by utterance id."""
    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                results[r["id"]] = r
    return results


def load_json_sidecar(ts: str) -> dict | None:
    """Load the JSON sidecar for a WAV timestamp."""
    path = AUDIO_DIR / f"{ts}.json"
    if not path.exists():
        return None
    try:
        content = path.read_text().strip()
        return json.loads(content)
    except json.JSONDecodeError:
        # Try stripping trailing braces
        for _ in range(3):
            if content.endswith("}"):
                content = content.rstrip().rstrip("}").rstrip()
                try:
                    return json.loads(content + "}")
                except json.JSONDecodeError:
                    continue
        return None


def play_wav(ts: str):
    """Play a WAV file via afplay. Returns after playback completes."""
    wav = AUDIO_DIR / f"{ts}.wav"
    if not wav.exists():
        print(f"  WAV not found: {wav}")
        return
    subprocess.run(["afplay", str(wav)], capture_output=True)


def main():
    results = load_results()
    print(f"Loaded {len(results)} benchmark results")

    # Get all WAVs that have JSON sidecars (paired)
    wavs = sorted(f.stem for f in AUDIO_DIR.glob("*.wav"))
    json_ids = {f.stem for f in AUDIO_DIR.glob("*.json")}

    start = None
    limit = None
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--start" and i + 1 < len(sys.argv):
            start = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    skipped = 0
    for ts in wavs:
        if start and ts < start:
            continue
        if limit is not None and skipped >= limit:
            break

        # Only review entries with JSON sidecars
        if ts not in json_ids:
            continue

        sidecar = load_json_sidecar(ts)
        if not sidecar:
            continue

        ref = sidecar.get("text", "").strip()
        result = results.get(ts)
        hyp = result["hyp_text"] if result else "(not yet benchmarked)"
        wer = f'WER={result["wer"]:.3f}' if result else ""

        print(f"\n{'='*60}")
        print(f"ID: {ts}")
        print(f"  REF:  \"{ref[:120]}\"")
        if result:
            print(f"  HYP:  \"{hyp[:120]}\"  [{wer}]")

        # Auto-skip clean entries (WER < 0.05)
        if result and result["wer"] < 0.05 and not result["hallucination"]:
            print("  -> SKIP (clean, WER < 5%)")
            skipped += 1
            continue

        # Play audio
        print("  -> Playing audio... (Ctrl+C to stop, Enter to skip)")
        try:
            play_wav(ts)
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            break

        # Prompt for correction
        action = input("  [Enter=skip, c=correct, q=quit]: ").strip().lower()
        if action == "q":
            break
        elif action == "c":
            new_text = input("  New text: ").strip()
            if new_text:
                sidecar["text"] = new_text
                sidecar["reviewed"] = True
                json_path = AUDIO_DIR / f"{ts}.json"
                json_path.write_text(json.dumps(sidecar, indent=2) + "\n")
                print(f"  -> Updated {json_path.name}")

        skipped += 1

    print(f"\nReviewed {skipped} entries.")
    print("Run 'python3 Scripts/build_corpus.py' to rebuild the corpus with corrections.")


if __name__ == "__main__":
    main()
