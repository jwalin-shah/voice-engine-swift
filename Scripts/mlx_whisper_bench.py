#!/usr/bin/env python3
"""
mlx_whisper_bench.py - Benchmark MLX Whisper against the voice-engine corpus.

MLX Whisper runs on Apple Silicon GPU via the MLX framework — the fastest
Whisper implementation available on Mac. Benchmarks both turbo and large-v3.

Usage:
    python3 Scripts/mlx_whisper_bench.py
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

import mlx_whisper
import soundfile as sf

CORPUS_PATH = Path(__file__).resolve().parent.parent / "bench_data" / "corpus.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "bench_data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {"id": "mlx-whisper-large-v3-turbo", "hf_repo": "mlx-community/whisper-large-v3-turbo"},
    {"id": "mlx-whisper-large-v3-mlx", "hf_repo": "mlx-community/whisper-large-v3-mlx"},
]


# --- Metrics (exact match with existing benchmarks: whisper_bench.py, faster_whisper_bench.py) ---

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


def get_audio_seconds(wav_path: str) -> float:
    """Read actual audio duration via soundfile (handles IEEE float WAVs)."""
    try:
        info = sf.info(wav_path)
        return info.duration
    except Exception:
        return 0.0


# --- Aggregate ---

def compute_aggregate(results: list, model_name: str) -> dict:
    """Compute aggregate stats from per-utterance results."""
    evaluable = [r for r in results if r.get("ref_text", "").strip()]
    n_total = len(evaluable)

    if n_total == 0:
        return {"model_name": model_name, "total_utterances": 0}

    by_cat = defaultdict(list)
    for r in evaluable:
        cat = r.get("category", "unknown")
        by_cat[cat].append(r)

    n_hallucinations = sum(1 for r in evaluable if r.get("hallucination", False))

    avg_wer = sum(r["wer"] for r in evaluable) / n_total
    avg_cer = sum(r["cer"] for r in evaluable) / n_total
    avg_rtf = sum(r["rtf"] for r in evaluable) / n_total
    avg_total_ms = sum(r["total_ms"] for r in evaluable) / n_total
    hallucination_rate = n_hallucinations / n_total

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
        "avg_total_ms": avg_total_ms,
        "avg_rtf": avg_rtf,
        "hallucination_rate": hallucination_rate,
        "by_category": by_category,
    }


# --- Main ---

def run_benchmark(model_config: dict):
    model_id = model_config["id"]
    hf_repo = model_config["hf_repo"]

    jsonl_path = RESULTS_DIR / f"{model_id}.jsonl"
    aggregate_path = RESULTS_DIR / f"{model_id}_aggregate.json"

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Model: {hf_repo}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # Load corpus — only entries with non-empty ref_text are evaluable
    corpus = []
    for line in open(CORPUS_PATH):
        entry = json.loads(line)
        if entry.get("ref_text", "").strip():
            corpus.append(entry)

    print(f"Evaluable utterances: {len(corpus)}", file=sys.stderr)

    # Warmup: transcribe first file to compile Metal shaders / load model
    if corpus:
        print(f"Warmup run...", file=sys.stderr)
        _ = mlx_whisper.transcribe(corpus[0]["wav_path"], path_or_hf_repo=hf_repo)

    results = []
    for idx, entry in enumerate(corpus):
        uid = entry["id"]
        wav_path = entry["wav_path"]
        ref_text = entry["ref_text"]
        category = entry.get("category", "unknown")

        if not os.path.exists(wav_path):
            print(f"  [{idx+1}/{len(corpus)}] {uid}: WAV not found, skipping", file=sys.stderr)
            results.append({
                "id": uid,
                "wav_path": wav_path,
                "ref_text": ref_text,
                "hyp_text": "",
                "category": category,
                "wer": 0,
                "cer": 0,
                "total_ms": 0,
                "audio_secs": 0,
                "rtf": 0,
                "hallucination": False,
            })
            continue

        audio_secs = get_audio_seconds(wav_path)

        t0 = time.time()
        try:
            result = mlx_whisper.transcribe(wav_path, path_or_hf_repo=hf_repo)
        except Exception as e:
            print(f"  [{idx+1}/{len(corpus)}] {uid}: ERROR: {e}", file=sys.stderr)
            results.append({
                "id": uid,
                "wav_path": wav_path,
                "ref_text": ref_text,
                "hyp_text": "",
                "category": category,
                "wer": 0,
                "cer": 0,
                "total_ms": 0,
                "audio_secs": audio_secs,
                "rtf": 0,
                "hallucination": False,
            })
            continue

        total_ms = (time.time() - t0) * 1000
        hyp_text = (result.get("text") or "").strip()

        # Compute metrics
        ref_words = normalize(ref_text)
        hyp_words = normalize(hyp_text)
        w = wer(ref_words, hyp_words)
        c = cer(ref_words, hyp_words)
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
        }
        results.append(record)

        print(f"  [{idx+1}/{len(corpus)}] {uid}  WER={w:.3f}  CER={c:.3f}  "
              f"total={total_ms:.0f}ms  RTF={rtf:.3f}  "
              f"{'HALLUCINATION' if hallucination else ''}",
              file=sys.stderr)

    # Save per-utterance results
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nPer-utterance results -> {jsonl_path}", file=sys.stderr)

    # Compute and save aggregate
    agg = compute_aggregate(results, model_id)
    with open(aggregate_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate results -> {aggregate_path}", file=sys.stderr)

    # Print summary
    print(f"\n══ {model_id} Summary ══", file=sys.stderr)
    print(f"  Utterances:     {agg['total_utterances']}", file=sys.stderr)
    print(f"  Avg WER:        {agg['avg_wer']:.4f}", file=sys.stderr)
    print(f"  Avg CER:        {agg['avg_cer']:.4f}", file=sys.stderr)
    print(f"  Avg total_ms:   {agg['avg_total_ms']:.1f}", file=sys.stderr)
    print(f"  Avg RTF:        {agg['avg_rtf']:.4f}", file=sys.stderr)
    print(f"  Hallucination:  {agg['hallucination_rate']:.2%}", file=sys.stderr)

    for cat_info in agg.get("by_category", []):
        print(f"    {cat_info['category']}: n={cat_info['count']}  "
              f"WER={cat_info['avg_wer']:.4f}  CER={cat_info['avg_cer']:.4f}  "
              f"RTF={cat_info['avg_rtf']:.4f}  hall={cat_info['hallucination_rate']:.2%}",
              file=sys.stderr)


if __name__ == "__main__":
    for cfg in MODELS:
        try:
            run_benchmark(cfg)
        except Exception as e:
            print(f"\nFAILED: {cfg['id']} — {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
