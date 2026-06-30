#!/usr/bin/env python3
"""Download 30 real LibriSpeech test-clean clips via huggingface_hub."""
import io, os, struct, sys, wave, random
from pathlib import Path

OUT = Path("/tmp/librispeech/test-clean")
N = 30

from huggingface_hub import hf_hub_download, HfApi
import pyarrow.parquet as pq

# Find a FLAC decoder
try:
    import soundfile as sf
    decoder = "soundfile"
except ImportError:
    try:
        import scipy.io.wavfile
        decoder = "scipy"
    except ImportError:
        decoder = None

if not decoder:
    print("Need soundfile or scipy. Installing soundfile...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "soundfile", "-q"])
    import soundfile as sf
    decoder = "soundfile"

random.seed(42)
OUT.mkdir(parents=True, exist_ok=True)

print(f"Downloading {N} LibriSpeech test-clean clips...")
api = HfApi()
files = api.list_repo_files("librispeech_asr", repo_type="dataset")
parquet_files = sorted(f for f in files if "clean/test" in f and f.endswith(".parquet"))

saved = 0
seen = set()

for pf in parquet_files:
    if saved >= N: break
    local = hf_hub_download("librispeech_asr", filename=pf, repo_type="dataset")
    table = pq.read_table(local, columns=["audio", "text", "speaker_id", "chapter_id", "id"])
    for i in range(len(table)):
        if saved >= N: break
        row = table.slice(i, 1)
        text = row["text"][0].as_py().strip().lower()
        if text in seen: continue
        seen.add(text)

        audio = row["audio"][0].as_py()
        flac_bytes = audio["bytes"]
        sid = str(row["speaker_id"][0].as_py())
        cid = str(row["chapter_id"][0].as_py())
        uid = str(row["id"][0].as_py())

        # Decode FLAC to PCM
        buf = io.BytesIO(flac_bytes)
        if decoder == "soundfile":
            arr, rate = sf.read(buf)
            if arr.ndim > 1: arr = arr.mean(axis=1)
        elif decoder == "scipy":
            import scipy.io.wavfile
            # scipy doesn't read FLAC natively, need to convert
            rate, arr = None, None
            continue

        subdir = OUT / sid / cid
        subdir.mkdir(parents=True, exist_ok=True)
        path = subdir / f"{sid}-{cid}-{uid}.wav"

        with wave.open(str(path), "w") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(rate))
            packed = b"".join(struct.pack("<h", int(s * 32767)) for s in arr)
            w.writeframes(packed)

        dur = len(arr) / rate
        clip = text[:55] + ("..." if len(text) > 55 else "")
        print(f"  {dur:5.1f}s  {path.name}  \"{clip}\"")
        saved += 1

print(f"\n✓ {saved} clips → {OUT}")
durs = []
for p in OUT.rglob("*.wav"):
    with wave.open(str(p)) as w:
        durs.append(w.getnframes() / w.getframerate())
if durs:
    print(f"  Range: {min(durs):.1f}s – {max(durs):.1f}s, mean {sum(durs)/len(durs):.1f}s")
