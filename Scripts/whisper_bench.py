#!/usr/bin/env python3
"""
whisper_bench.py - Benchmark whisper.cpp against the voice-engine corpus.

Reads bench_data/corpus.jsonl, transcribes each entry with ref_text,
computes WER/CER/hallucination, outputs results to bench_data/results/.

Usage:
    python3 Scripts/whisper_bench.py [--model small.en] [--model-path /path/to/model.bin]
"""
import json, re, subprocess, sys, tempfile, os, time, argparse
from pathlib import Path
from collections import defaultdict

WHISPER_CLI = "/tmp/whisper.cpp/build-coreml/bin/whisper-cli"
MODEL_DIR = "/tmp/whisper.cpp/models"
CORPUS_PATH = Path(__file__).resolve().parent.parent / "bench_data" / "corpus.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "bench_data" / "results"


# --- Metrics (from sweep.py:25-48) ---

def normalize(text: str) -> list:
    """LibriSpeech-style normalization: uppercase, strip punctuation, split."""
    text = text.upper()
    text = re.sub(r"[^A-Z' ]", " ", text)
    return [w for w in text.split() if w]


def levenshtein(ref: list, hyp: list) -> int:
    """Levenshtein distance (edit count). Returns raw distance, not ratio."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def wer(ref_words: list, hyp_words: list) -> float:
    """Word error rate via Levenshtein on word lists."""
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


def cer(ref_words: list, hyp_words: list) -> float:
    """Character error rate via Levenshtein on space-joined character lists."""
    ref_chars = list(" ".join(ref_words))
    hyp_chars = list(" ".join(hyp_words))
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def is_hallucination(ref_words: list, hyp_words: list) -> bool:
    """Hypothesis >3x ref length or no overlapping words."""
    if len(hyp_words) > 3 * max(len(ref_words), 1):
        return True
    ref_set = set(ref_words)
    hyp_set = set(hyp_words)
    if len(ref_set) > 0 and len(ref_set & hyp_set) == 0:
        return True
    return False


# --- Transcribe ---

def run_whisper(wav_path: str, model_path: str) -> tuple:
    """Run whisper-cli on a WAV file. Returns (text, timing_dict)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_prefix = tmp.name.replace(".json", "")

    try:
        cmd = [
            WHISPER_CLI,
            "-m", model_path,
            "-f", wav_path,
            "--no-timestamps",
            "-oj",
            "-of", tmp_prefix,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stderr = result.stderr

        # Parse JSON output for text
        json_path = tmp_prefix + ".json"
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
        return text, timing

    finally:
        for p in [tmp.name, tmp_prefix + ".json"]:
            if Path(p).exists():
                try:
                    os.unlink(p)
                except OSError:
                    pass


def parse_timing(stderr: str) -> dict:
    """Parse whisper_print_timings from stderr."""
    timing = {}
    patterns = {
        "load_ms": r"load time\s*=\s*([\d.]+)\s*ms",
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
    return timing


# --- Aggregate ---

def compute_aggregate(results: list, model_name: str) -> dict:
    """Compute aggregate stats from per-utterance results."""
    by_cat = defaultdict(list)
    for r in results:
        cat = r.get("category", "unknown")
        by_cat[cat].append(r)

    n_total = len(results)
    n_hallucinations = sum(1 for r in results if r.get("hallucination", False))
    if n_total > 0:
        avg_wer = sum(r["wer"] for r in results) / n_total
        avg_cer = sum(r["cer"] for r in results) / n_total
        avg_rtf = sum(r["rtf"] for r in results) / n_total
        avg_total_ms = sum(r["total_ms"] for r in results) / n_total
        hallucination_rate = n_hallucinations / n_total
    else:
        avg_wer = avg_cer = avg_rtf = avg_total_ms = hallucination_rate = 0

    by_category = []
    for cat in sorted(by_cat.keys()):
        rs = by_cat[cat]
        n = len(rs)
        by_category.append({
            "category": cat,
            "count": n,
            "avg_wer": sum(r["wer"] for r in rs) / n if n else 0,
            "avg_cer": sum(r["cer"] for r in rs) / n if n else 0,
            "avg_rtf": sum(r["rtf"] for r in rs) / n if n else 0,
            "hallucination_rate": sum(1 for r in rs if r.get("hallucination", False)) / n if n else 0,
        })

    return {
        "model_name": model_name,
        "total_utterances": n_total,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "avg_rtf": avg_rtf,
        "avg_total_ms": avg_total_ms,
        "hallucination_rate": hallucination_rate,
        "by_category": by_category,
    }


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Benchmark whisper.cpp against voice-engine corpus")
    parser.add_argument("--model", default="small.en", help="Model name (default: small.en)")
    parser.add_argument("--model-path", default=None, help="Full path to model file")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N utterances (0=all)")
    args = parser.parse_args()

    model_name = args.model
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = f"{MODEL_DIR}/ggml-{model_name}.bin"

    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not Path(WHISPER_CLI).exists():
        print(f"ERROR: whisper-cli not found: {WHISPER_CLI}", file=sys.stderr)
        sys.exit(1)

    # Load corpus
    corpus = []
    for line in open(CORPUS_PATH):
        entry = json.loads(line)
        if entry.get("ref_text"):
            corpus.append(entry)

    if args.limit > 0:
        corpus = corpus[:args.limit]

    print(f"Benchmarking {model_name} on {len(corpus)} utterances with ref_text", file=sys.stderr)
    print(f"Model: {model_path}", file=sys.stderr)

    # Warmup: transcribe first file to compile Metal shaders
    if corpus:
        print(f"Warmup run...", file=sys.stderr)
        _ = run_whisper(corpus[0]["wav_path"], model_path)

    # Transcribe each utterance
    results = []
    results_path = RESULTS_DIR / f"whisper-{model_name}.jsonl"

    for idx, entry in enumerate(corpus):
        uid = entry["id"]
        wav_path = entry["wav_path"]
        ref_text = entry["ref_text"]
        category = entry.get("category", "unknown")

        if not Path(wav_path).exists():
            print(f"  [{idx+1}/{len(corpus)}] {uid}: WAV not found, skipping", file=sys.stderr)
            continue

        t0 = time.perf_counter()
        try:
            hyp_text, timing = run_whisper(wav_path, model_path)
        except subprocess.TimeoutExpired:
            print(f"  [{idx+1}/{len(corpus)}] {uid}: TIMEOUT", file=sys.stderr)
            continue
        except Exception as e:
            print(f"  [{idx+1}/{len(corpus)}] {uid}: ERROR: {e}", file=sys.stderr)
            continue
        wall_ms = (time.perf_counter() - t0) * 1000

        # Compute metrics
        ref_words = normalize(ref_text)
        hyp_words = normalize(hyp_text)
        w = wer(ref_words, hyp_words)
        c = cer(ref_words, hyp_words)

        # Get audio duration
        try:
            import soundfile as sf
            info = sf.info(wav_path)
            audio_secs = info.duration
        except Exception:
            audio_secs = None

        # Processing time (exclude model loading)
        total_ms = timing.get("total_ms", 0) - timing.get("load_ms", 0)
        rtf = total_ms / (audio_secs * 1000) if audio_secs and audio_secs > 0 else 0

        hallucination = is_hallucination(ref_words, hyp_words)

        record = {
            "id": uid,
            "wav_path": wav_path,
            "ref_text": ref_text,
            "hyp_text": hyp_text,
            "category": category,
            "wer": round(w, 6),
            "cer": round(c, 6),
            "rtf": round(rtf, 6),
            "hallucination": hallucination,
            "audio_secs": round(audio_secs, 6) if audio_secs else None,
            "total_ms": round(total_ms, 2),
            "wall_ms": round(wall_ms, 2),
        }
        results.append(record)

        print(f"  [{idx+1}/{len(corpus)}] {uid}  WER={w:.3f}  CER={c:.3f}  "
              f"total={total_ms:.0f}ms  RTF={rtf:.3f}  "
              f"{'HALLUCINATION' if hallucination else ''}",
              file=sys.stderr)

    # Save per-utterance results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} results to {results_path}", file=sys.stderr)

    # Compute and save aggregate
    agg = compute_aggregate(results, f"whisper-{model_name}")
    agg_path = RESULTS_DIR / f"whisper-{model_name}_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Saved aggregate to {agg_path}", file=sys.stderr)

    # Print summary
    print(f"\n══ whisper-{model_name} Summary ══", file=sys.stderr)
    print(f"  Utterances:     {agg['total_utterances']}", file=sys.stderr)
    print(f"  Avg WER:        {agg['avg_wer']:.4f}", file=sys.stderr)
    print(f"  Avg CER:        {agg['avg_cer']:.4f}", file=sys.stderr)
    print(f"  Avg total_ms:   {agg['avg_total_ms']:.1f}", file=sys.stderr)
    print(f"  Avg RTF:        {agg['avg_rtf']:.4f}", file=sys.stderr)
    print(f"  Hallucination:  {agg['hallucination_rate']:.2%}", file=sys.stderr)

    for cat_info in agg["by_category"]:
        print(f"    {cat_info['category']}: n={cat_info['count']}  "
              f"WER={cat_info['avg_wer']:.4f}  CER={cat_info['avg_cer']:.4f}  "
              f"RTF={cat_info['avg_rtf']:.4f}  hall={cat_info['hallucination_rate']:.2%}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
