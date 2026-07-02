#!/usr/bin/env python3
"""Run benchmark on LibriSpeech test-clean clips with WER + per-bucket timing."""
import argparse, json, sys, subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BENCH = REPO_ROOT / "bench.py"


def normalize_words(text):
    return "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()


def word_error_counts(reference, hypothesis):
    """Return (edit_distance, reference_word_count) for WER aggregation."""
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    if not ref:
        return (0 if not hyp else len(hyp)), len(ref)

    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1], len(ref)

def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wav-dir",
        type=Path,
        default=Path("/tmp/librispeech/test-clean"),
        help="Directory containing LibriSpeech test-clean WAV files",
    )
    ap.add_argument("--limit", type=positive_int, default=30, help="Maximum number of clips to benchmark")
    return ap.parse_args()


def load_refs():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    local = hf_hub_download("librispeech_asr", filename="clean/test/0000.parquet", repo_type="dataset")
    table = pq.read_table(local, columns=["audio", "text", "speaker_id", "chapter_id", "id"])
    refs = {}
    for i in range(len(table)):
        row = table.slice(i, 1)
        sid = str(row["speaker_id"][0].as_py())
        cid = str(row["chapter_id"][0].as_py())
        utterance_id = str(row["id"][0].as_py())
        text = row["text"][0].as_py()
        refs[utterance_id] = text
        refs[f"{sid}-{cid}-{utterance_id}"] = text
    return refs


def run_bench(wavs, refs):
    results = []
    for wav in wavs:
        result = run_one(wav, refs)
        if result:
            results.append(result)
    return results


def run_one(wav, refs):
    stem = wav.stem
    ref = refs.get(stem)
    if ref is None:
        print(f"  ! {stem}: no LibriSpeech reference found; skipping")
        return None

    try:
        result = subprocess.run(
            [sys.executable, str(BENCH), str(wav), "--iterations", "1", "--json"],
            capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        print(f"  ! {stem}: timed out after 300s; skipping")
        return None
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
        return None

    dur = data.get("audio_s", 0)
    hyp = data.get("text", "")
    total = sum(data.get(k, {}).get("mean_ms", 0)
                for k in ["preprocess","encoder","kv_projection","decoder_loop","token_decode"])

    ref_norm = " ".join(normalize_words(ref))
    hyp_norm = " ".join(normalize_words(hyp))
    wer_errors, ref_words = word_error_counts(ref, hyp)
    wer_pct = (100 * wer_errors / ref_words) if ref_words else (0.0 if not hyp_norm else 100.0)
    match = "✓" if wer_errors == 0 else "✗"

    bucket = "0-2s" if dur <= 2 else "2-5s" if dur <= 5 else "5-10s" if dur <= 10 else "10-15s" if dur <= 15 else "15s+"

    enc_ms = data.get("encoder", {}).get("mean_ms", 0)
    kv_ms = data.get("kv_projection", {}).get("mean_ms", 0)
    dec_ms = data.get("decoder_loop", {}).get("mean_ms", 0)
    steps = data.get("decoder_steps", 0)

    print(f"  {match} {dur:5.1f}s  [{bucket:6s}] "
          f"enc={enc_ms:4.1f} kv={kv_ms:4.1f} dec={dec_ms:5.1f} ({steps:3d} steps) "
          f"total={total:5.1f}ms wer={wer_pct:5.1f}%")
    if match == "✗":
        print(f"       ref: \"{ref_norm[:60]}\"")
        print(f"       hyp: \"{hyp_norm[:60]}\"")

    return {"file": stem, "dur": dur, "bucket": bucket,
            "enc_ms": enc_ms, "kv_ms": kv_ms, "dec_ms": dec_ms,
            "steps": steps, "total_ms": total,
            "match": match == "✓", "wer_errors": wer_errors,
            "ref_words": ref_words, "wer_pct": wer_pct,
            "ref": ref_norm, "hyp": hyp_norm}


def print_summary(results):
    print(f"\n{'='*60}")
    if not results:
        print("No successful benchmark results were collected.")
        print("Check the errors above; common causes are missing CoreML models or failed bench.py runs.")
        sys.exit(1)

    correct = sum(1 for r in results if r["match"])
    total_errors = sum(r["wer_errors"] for r in results)
    total_ref_words = sum(r["ref_words"] for r in results)
    corpus_wer = (100 * total_errors / total_ref_words) if total_ref_words else 0.0
    print(f"Exact transcripts: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"Corpus WER: {corpus_wer:.1f}%")
    print(f"Mean total: {sum(r['total_ms'] for r in results)/len(results):.1f} ms")

    buckets = {}
    for r in results:
        b = r["bucket"]
        if b not in buckets:
            buckets[b] = {
                "n": 0, "total": 0, "enc": 0, "kv": 0, "dec": 0, "steps": 0,
                "correct": 0, "wer_errors": 0, "ref_words": 0,
            }
        buckets[b]["n"] += 1
        buckets[b]["total"] += r["total_ms"]
        buckets[b]["enc"] += r["enc_ms"]
        buckets[b]["kv"] += r["kv_ms"]
        buckets[b]["dec"] += r["dec_ms"]
        buckets[b]["steps"] += r["steps"]
        buckets[b]["correct"] += 1 if r["match"] else 0
        buckets[b]["wer_errors"] += r["wer_errors"]
        buckets[b]["ref_words"] += r["ref_words"]

    print(f"\n{'Bucket':<10} {'n':>3} {'Enc':>6} {'KV':>6} {'Dec':>8} {'Steps':>5} {'Total':>6} {'WER':>6}")
    print("-" * 50)
    for b in sorted(buckets.keys()):
        v = buckets[b]
        n = v["n"]
        wer = (100 * v["wer_errors"] / v["ref_words"]) if v["ref_words"] else 0.0
        print(f"{b:<10} {n:>3} {v['enc']/n:5.1f}ms {v['kv']/n:5.1f}ms {v['dec']/n:5.1f}ms {v['steps']/n:4.0f} {v['total']/n:5.1f}ms {wer:5.1f}%")

    with open("/tmp/libri_bench_results.json", "w") as f:
        json.dump({"results": results, "summary": {
            "correct": correct, "total": len(results),
            "corpus_wer_pct": corpus_wer,
            "mean_total_ms": sum(r['total_ms'] for r in results)/len(results)
        }}, f, indent=2)
    print(f"\nResults saved to /tmp/libri_bench_results.json")


def main():
    args = parse_args()
    wavs = sorted(args.wav_dir.rglob("*.wav"))[:args.limit]
    print(f"Found {len(wavs)} WAV files\n")
    if not wavs:
        print(f"No LibriSpeech WAV files found in {args.wav_dir}")
        print("Run fetch_librispeech.py first, or pass --wav-dir with a populated test-clean directory.")
        sys.exit(1)

    refs = load_refs()
    results = run_bench(wavs, refs)
    print_summary(results)


if __name__ == "__main__":
    main()
