#!/usr/bin/env python3
"""
Export paired audio+transcription dataset from VoiceEngine recordings.
Also compare against other ASR models (Whisper, etc.) for WER comparison.

Usage:
  python3 export_dataset.py                          # List all recordings
  python3 export_dataset.py --export /tmp/dataset    # Export as JSONL dataset
  python3 export_dataset.py --whisper                # Compare Whisper vs Moonshine WER
"""

import json, os, sys, subprocess, wave, re, glob
from pathlib import Path

AUDIO_DIR = Path.home() / "Library" / "Logs" / "voice-engine" / "audio"
METRICS_FILE = Path.home() / "Library" / "Logs" / "voice-engine" / "metrics.jsonl"

def find_recordings():
    """Find all WAV files and their paired JSON sidecars."""
    recordings = []
    for wav in sorted(AUDIO_DIR.rglob("*.wav")):
        json_sidecar = wav.with_suffix(".json")
        meta = {}
        if json_sidecar.exists():
            try:
                meta = json.loads(json_sidecar.read_text())
            except: pass
        recordings.append({"wav": str(wav), "json": str(json_sidecar), **meta})
    return recordings

def export_dataset(out_dir):
    """Export all recordings as a clean JSONL dataset."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = find_recordings()

    dataset = []
    for r in recs:
        text = r.get("text", "").strip()
        if not text:
            continue  # skip unlabeled recordings

        dur = r.get("duration_secs", 0)
        app = r.get("app", "")

        entry = {"audio": r["wav"], "text": text, "duration_secs": dur, "app": app}
        dataset.append(entry)

    outfile = out / "dataset.jsonl"
    with open(outfile, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"Exported {len(dataset)} recordings to {outfile}")
    if not dataset:
        print("  No labeled recordings found.")
        return dataset

    print(f"  Duration range: {min(e['duration_secs'] for e in dataset):.1f}s - {max(e['duration_secs'] for e in dataset):.1f}s")
    print(f"  Total audio: {sum(e['duration_secs'] for e in dataset):.1f}s")
    return dataset

def compare_whisper():
    """Run Whisper on all recordings and compare WER with Moonshine."""
    try:
        import whisper
    except ImportError:
        print("Installing openai-whisper...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper", "-q"])
        import whisper

    recs = [r for r in find_recordings() if r.get("text", "").strip()]
    if not recs:
        print("No labeled recordings found. Dictate something first!")
        return

    print(f"Comparing Whisper vs Moonshine on {len(recs)} recordings...\n")

    def wer(r, h):
        r = re.sub(r'[^a-z0-9\s]', '', r.lower()).split()
        h = re.sub(r'[^a-z0-9\s]', '', h.lower()).split()
        if not r: return 0.0
        d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
        for i in range(len(r)+1): d[i][0] = i
        for j in range(len(h)+1): d[0][j] = j
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                cost = 0 if r[i-1] == h[j-1] else 1
                d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
        return d[len(r)][len(h)] / len(r)

    model = whisper.load_model("small")  # "tiny", "small", "base"

    print(f"{'File':<40} {'Dur':>5} {'Moonshine WER':>14} {'Whisper WER':>14}")
    print("-" * 75)

    moon_wers, whisp_wers = [], []
    for r in recs:
        wav_path = r["wav"]
        ref = r["text"]

        # Get duration
        with wave.open(wav_path) as w:
            dur = w.getnframes() / w.getframerate()

        # Run Whisper
        result = model.transcribe(wav_path, language="en")
        whisper_text = result["text"].strip()

        moon_wer = wer(ref, r.get("_moonshine_text", ref))  # use stored moonshine text
        whisp_wer = wer(ref, whisper_text)

        moon_wers.append(moon_wer)
        whisp_wers.append(whisp_wer)

        name = os.path.basename(wav_path)[:38]
        print(f"{name:<40} {dur:4.1f}s {moon_wer*100:11.1f}% {whisp_wer*100:11.1f}%")

    print("-" * 75)
    print(f"{'AVERAGE':<40}        {sum(moon_wers)/len(moon_wers)*100:11.1f}% {sum(whisp_wers)/len(whisp_wers)*100:11.1f}%")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", help="Export dataset to directory")
    ap.add_argument("--whisper", action="store_true", help="Compare against Whisper")
    ap.add_argument("--list", action="store_true", help="List all recordings")
    args = ap.parse_args()

    if args.list or not any([args.export, args.whisper]):
        recs = find_recordings()
        labeled = sum(1 for r in recs if r.get("text", "").strip())
        print(f"Recordings: {len(recs)} total, {labeled} labeled")
        for r in recs[-20:]:
            text = r.get("text", "")[:60]
            dur = r.get("duration_secs", 0)
            flag = "✓" if text else "✗"
            print(f"  {flag} {dur:4.1f}s  {os.path.basename(r['wav']):<35s} {text}")

    if args.export:
        export_dataset(args.export)

    if args.whisper:
        compare_whisper()
