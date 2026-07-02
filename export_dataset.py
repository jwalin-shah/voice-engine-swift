#!/usr/bin/env python3
"""
Export paired audio+transcription dataset from VoiceEngine recordings.
Also compare against other ASR models (Whisper, etc.) for WER comparison.

Usage:
  python3 export_dataset.py                          # List all recordings
  python3 export_dataset.py --export /tmp/dataset    # Export as JSONL dataset
  python3 export_dataset.py --whisper                # Compare Whisper vs Moonshine WER
"""

import json, os, sys, subprocess, re, glob
from pathlib import Path

AUDIO_DIR = Path.home() / "Library" / "Logs" / "voice-engine" / "audio"
METRICS_FILE = Path.home() / "Library" / "Logs" / "voice-engine" / "metrics.jsonl"

def audio_duration_s(wav_path):
    """Return duration for supported VoiceEngine WAV archives."""
    from bench import inspect_audio_wav

    _, duration = inspect_audio_wav(str(wav_path))
    return duration

def recording_duration_s(recording):
    """Prefer sidecar duration, falling back to the actual WAV duration."""
    try:
        duration = float(recording.get("duration_secs", 0) or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration > 0:
        return duration

    try:
        return audio_duration_s(recording["wav"])
    except (FileNotFoundError, ValueError):
        return duration

def recording_text(recording):
    """Return a valid transcript string from recording metadata."""
    text = recording.get("text", "")
    return text.strip() if isinstance(text, str) else ""

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
        text = recording_text(r)
        if not text:
            continue  # skip unlabeled recordings

        dur = recording_duration_s(r)
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
    recs = [r for r in find_recordings() if recording_text(r)]
    if not recs:
        print("No labeled recordings found. Dictate something first!")
        return

    valid_recs = []
    for r in recs:
        try:
            r["_duration_s"] = audio_duration_s(r["wav"])
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping unsupported recording {os.path.basename(r['wav'])}: {exc}")
            continue
        valid_recs.append(r)

    if not valid_recs:
        print("No supported labeled WAV recordings found.")
        return

    try:
        import whisper
    except ImportError:
        print("Installing openai-whisper...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper", "-q"])
        import whisper

    recs = valid_recs
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
        ref = recording_text(r)

        dur = r["_duration_s"]

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
        labeled = sum(1 for r in recs if recording_text(r))
        print(f"Recordings: {len(recs)} total, {labeled} labeled")
        for r in recs[-20:]:
            text = recording_text(r)[:60]
            dur = recording_duration_s(r)
            flag = "✓" if text else "✗"
            print(f"  {flag} {dur:4.1f}s  {os.path.basename(r['wav']):<35s} {text}")

    if args.export:
        export_dataset(args.export)

    if args.whisper:
        compare_whisper()
