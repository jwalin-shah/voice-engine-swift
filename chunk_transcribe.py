#!/usr/bin/env python3
"""
Efficient chunked transcriber: loads models ONCE, then chunks through audio.
Overlap boundaries are handled by trimming the overlapping portion of the text.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

SR = 16000
CHUNK_SAMPLES = 10 * SR
OVERLAP_SAMPLES = 2 * SR  # 2s overlap
MODEL_DIR = Path.home() / ".cache" / "moonshine-coreml" / "tiny-streaming"
REPO_ROOT = Path(__file__).resolve().parent

# Import the benchmark's MoonshineBench
sys.path.insert(0, str(REPO_ROOT))
from bench import MoonshineBench, inspect_audio_wav, load_audio_wav

def inspect_wav(path):
    """Validate WAV metadata and return (frame_count, duration_s)."""
    return inspect_audio_wav(str(path))

def load_audio(path):
    samples = load_audio_wav(str(path))
    return samples, len(samples) / SR

def dedup_overlap(prev_text, new_text):
    """Sentence-level dedup: drop sentences from new_text that already appeared."""
    import re
    prev_sents = re.split(r'(?<=[.!?])\s+', prev_text.strip())
    new_sents = re.split(r'(?<=[.!?])\s+', new_text.strip())
    if not prev_sents or not new_sents:
        return new_text

    # Normalize: lowercase + strip punctuation
    def norm(s):
        return s.strip().lower().rstrip(".,!?;")

    # Get last 1-2 sentences of previous chunk
    tail = [norm(s) for s in (prev_sents[-2:] if len(prev_sents) >= 2 else prev_sents[-1:])]

    # Drop leading new sentences that match
    for skip in range(min(len(new_sents), 4)):
        h = norm(new_sents[skip])
        for t in tail:
            if not h or not t: continue
            # Match on significant overlap: 15+ chars identical, or one contains the other
            if (len(t) > 8 and len(h) > 8 and
                (h.startswith(t) or t.startswith(h) or
                 t[:15] == h[:15] or
                 t in h or h in t)):
                return ' '.join(new_sents[skip+1:]).strip()
        # Short fragments at start are always overlap
        if len(new_sents[skip].split()) <= 4:
            continue
        break

    return new_text

def transcribe_long(bench, path, overlap_s=2.0):
    """Transcribe using loaded model, chunk by chunk."""
    if not math.isfinite(overlap_s) or overlap_s < 0 or overlap_s >= CHUNK_SAMPLES / SR:
        raise ValueError(f"overlap_s must be finite, >= 0, and < {CHUNK_SAMPLES / SR:.1f}s")

    samples, dur = load_audio(path)
    n_samples = len(samples)
    step = CHUNK_SAMPLES - int(overlap_s * SR)

    # Pad to full chunk
    if n_samples < CHUNK_SAMPLES:
        samples = np.pad(samples, (0, CHUNK_SAMPLES - n_samples)).astype(np.float32, copy=False)
        t0 = time.perf_counter()
        text, _ = bench.transcribe(samples)
        decode_ms = (time.perf_counter() - t0) * 1000
        return text.strip(), decode_ms

    full_text = ""
    prev_text = ""
    total_decode_ms = 0
    previous_end = 0

    for start in range(0, n_samples, step):
        end = min(start + CHUNK_SAMPLES, n_samples)
        new_audio_samples = end - previous_end
        if start > 0 and new_audio_samples < SR:
            break
        if end - start < SR:
            break

        chunk = samples[start:end]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

        t0 = time.perf_counter()
        audio_np = np.asarray(chunk, dtype=np.float32)
        text, times = bench.transcribe(audio_np)
        decode_ms = (time.perf_counter() - t0) * 1000
        total_decode_ms += decode_ms

        # Clean up the text
        text = text.strip()

        # Remove overlap with previous chunk
        if prev_text:
            new_part = dedup_overlap(prev_text, text)
        else:
            new_part = text

        if new_part:
            full_text += " " + new_part

        t_s = start / SR
        t_e = end / SR
        clip = text[:50].replace("\n", " ")
        print(f"  chunk {start//step:2d} ({t_s:.0f}s-{t_e:.0f}s): "
              f"{decode_ms:5.0f}ms \"{clip}...\"")

        prev_text = text
        previous_end = end

    return full_text.strip(), total_decode_ms

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Transcribe a 16 kHz mono WAV with the Moonshine CoreML chunking pipeline. "
            "Supports 16-bit PCM and 32-bit float WAV input."
        )
    )
    ap.add_argument("audio", nargs="?", default="/tmp/libri_chained.wav")
    ap.add_argument("--overlap", type=float, default=2.0, help="Chunk overlap in seconds")
    args = ap.parse_args()

    path = Path(args.audio)
    if not math.isfinite(args.overlap) or args.overlap < 0 or args.overlap >= CHUNK_SAMPLES / SR:
        print(f"ERROR: --overlap must be finite, >= 0, and < {CHUNK_SAMPLES / SR:.1f}s", file=sys.stderr)
        sys.exit(2)
    try:
        _, dur = inspect_wav(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    print("Loading model (one time)...")
    bench = MoonshineBench()
    if not bench.load():
        sys.exit(1)
    bench.warmup()

    print(f"\nTranscribing {path}...")
    text, ms = transcribe_long(bench, path, overlap_s=args.overlap)

    words = len(text.split())

    print(f"\n{'='*60}")
    print(f"Duration: {dur:.0f}s | Decode: {ms:.0f}ms | RTF: {ms/1000/dur:.5f}")
    print(f"Words: {words} ({words/(dur/60):.0f} wpm)")
    print(f"{'='*60}")
    print(text)
