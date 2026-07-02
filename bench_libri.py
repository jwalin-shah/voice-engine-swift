#!/usr/bin/env python3
"""Run benchmark on all LibriSpeech test-clean clips with WER + per-bucket timing."""
import json, os, sys, subprocess, wave
from pathlib import Path

WAV_DIR = Path("/tmp/librispeech/test-clean")
REPO_ROOT = Path(__file__).resolve().parent
BENCH = REPO_ROOT / "bench.py"
N = 30

# Collect all WAVs and their transcripts from filenames + parquet
wavs = sorted(WAV_DIR.rglob("*.wav"))
print(f"Found {len(wavs)} WAV files\n")
if not wavs:
    print(f"No LibriSpeech WAV files found in {WAV_DIR}")
    print("Run fetch_librispeech.py first, or set up /tmp/librispeech/test-clean.")
    sys.exit(1)

# Load reference transcripts from parquet
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
local = hf_hub_download("librispeech_asr", filename="clean/test/0000.parquet", repo_type="dataset")
table = pq.read_table(local, columns=["audio", "text", "speaker_id", "chapter_id", "id"])
# Build transcript lookup by id
refs = {}
for i in range(len(table)):
    row = table.slice(i, 1)
    rid = f"{row['speaker_id'][0].as_py()}-{row['chapter_id'][0].as_py()}-{row['id'][0].as_py()}"
    refs[rid] = row["text"][0].as_py()

# Run each clip through the benchmark
results = []
for wav in wavs[:N]:
    stem = wav.stem
    ref = refs.get(stem, "?")

    result = subprocess.run(
        [sys.executable, str(BENCH), str(wav), "--iterations", "1", "--json"],
        capture_output=True, text=True, timeout=300
    )
    # Show any errors
    if result.returncode != 0:
        stderr_short = result.stderr.strip()[:200] if result.stderr else "none"
        print(f"  ! {stem}: exit={result.returncode}, stderr: {stderr_short}")

    # Extract JSON from output — find first '{' and parse from there
    data = None
    start = result.stdout.find("{\n")
    if start >= 0:
        try:
            data = json.loads(result.stdout[start:])
        except json.JSONDecodeError:
            pass

    if data is None:
        print(f"  ✗ {stem}: no JSON output")
        continue

    dur = data.get("audio_s", 0)
    hyp = data.get("text", "")
    total = sum(data.get(k, {}).get("mean_ms", 0)
                for k in ["preprocess","encoder","kv_projection","decoder_loop","token_decode"])

    # Simple WER: just check if hyp matches ref (case-insensitive)
    ref_norm = ref.strip().lower()
    hyp_norm = hyp.strip().lower()
    match = "✓" if hyp_norm == ref_norm else "✗"

    bucket = "0-2s" if dur <= 2 else "2-5s" if dur <= 5 else "5-10s" if dur <= 10 else "10-15s" if dur <= 15 else "15s+"

    enc_ms = data.get("encoder", {}).get("mean_ms", 0)
    kv_ms = data.get("kv_projection", {}).get("mean_ms", 0)
    dec_ms = data.get("decoder_loop", {}).get("mean_ms", 0)
    steps = data.get("decoder_steps", 0)

    print(f"  {match} {dur:5.1f}s  [{bucket:6s}] "
          f"enc={enc_ms:4.1f} kv={kv_ms:4.1f} dec={dec_ms:5.1f} ({steps:3d} steps) "
          f"total={total:5.1f}ms")
    if match == "✗":
        print(f"       ref: \"{ref_norm[:60]}\"")
        print(f"       hyp: \"{hyp_norm[:60]}\"")

    results.append({"file": stem, "dur": dur, "bucket": bucket,
                     "enc_ms": enc_ms, "kv_ms": kv_ms, "dec_ms": dec_ms,
                     "steps": steps, "total_ms": total,
                     "match": match == "✓", "ref": ref_norm, "hyp": hyp_norm})

# Summary
print(f"\n{'='*60}")
if not results:
    print("No successful benchmark results were collected.")
    print("Check the errors above; common causes are missing CoreML models or failed bench.py runs.")
    sys.exit(1)

correct = sum(1 for r in results if r["match"])
print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
print(f"Mean total: {sum(r['total_ms'] for r in results)/len(results):.1f} ms")

buckets = {}
for r in results:
    b = r["bucket"]
    if b not in buckets: buckets[b] = {"n":0, "total":0, "enc":0, "kv":0, "dec":0, "steps":0, "correct":0}
    buckets[b]["n"] += 1
    buckets[b]["total"] += r["total_ms"]
    buckets[b]["enc"] += r["enc_ms"]
    buckets[b]["kv"] += r["kv_ms"]
    buckets[b]["dec"] += r["dec_ms"]
    buckets[b]["steps"] += r["steps"]
    buckets[b]["correct"] += 1 if r["match"] else 0

print(f"\n{'Bucket':<10} {'n':>3} {'Enc':>6} {'KV':>6} {'Dec':>8} {'Steps':>5} {'Total':>6} {'WER':>6}")
print("-" * 50)
for b in sorted(buckets.keys()):
    v = buckets[b]
    n = v["n"]
    wer = (1 - v["correct"]/n)*100 if n else 0
    print(f"{b:<10} {n:>3} {v['enc']/n:5.1f}ms {v['kv']/n:5.1f}ms {v['dec']/n:5.1f}ms {v['steps']/n:4.0f} {v['total']/n:5.1f}ms {wer:5.1f}%")

# Save results
with open("/tmp/libri_bench_results.json", "w") as f:
    json.dump({"results": results, "summary": {
        "correct": correct, "total": len(results),
        "mean_total_ms": sum(r['total_ms'] for r in results)/len(results)
    }}, f, indent=2)
print(f"\nResults saved to /tmp/libri_bench_results.json")
