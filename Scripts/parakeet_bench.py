#!/usr/bin/env python3
"""
Benchmark Parakeet-TDT-0.6B (NeMo EncDecRNNTBPEModel) against the voice-engine corpus.

Matches the WER/CER/metrics from Sources/VoiceEngine/Benchmark.swift exactly.
"""
import json
import os
import re
import time
from pathlib import Path
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

CORPUS_PATH = Path(__file__).resolve().parent.parent / "bench_data" / "corpus.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "bench_data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NEMO_PATH = "/tmp/parakeet/parakeet-tdt-0.6b-v2.nemo"
MODEL_ID = "parakeet-tdt-0.6b"
MODEL_NAME = "Parakeet-TDT-0.6B (CPU)"


# -- Metrics (match Benchmark.swift exactly) --

def normalize(text: str) -> list[str]:
    """LibriSpeech-style: uppercase, strip punctuation, split on whitespace."""
    text = text.upper()
    text = re.sub(r"[^A-Z' ]", " ", text)
    return [w for w in text.split() if w]


def wer_details(ref: list[str], hyp: list[str]) -> tuple[float, float, float, float]:
    """Returns (wer, deletions, insertions, substitutions) as fractions of ref length."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return (0.0 if m == 0 else 1.0, 0.0, float(m) / max(1, m), 0.0)
    if m == 0:
        return (1.0, 1.0, 0.0, 0.0)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = (
                dp[i - 1][j - 1]
                if ref[i - 1] == hyp[j - 1]
                else 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            )
    i, j = n, m
    deletions = 0
    insertions = 0
    substitutions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    wer_val = dp[n][m] / n
    return (wer_val, deletions / n, insertions / n, substitutions / n)


def cer(ref_chars: list[str], hyp_chars: list[str]) -> float:
    """Character error rate matching Benchmark.cer."""
    n, m = len(ref_chars), len(hyp_chars)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    if m == 0:
        return 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = (
                dp[i - 1][j - 1]
                if ref_chars[i - 1] == hyp_chars[j - 1]
                else 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            )
    return dp[n][m] / n


def get_audio_seconds(wav_path: str) -> float:
    """Read actual PCM duration from WAV file."""
    try:
        info = sf.info(wav_path)
        return info.duration
    except Exception:
        return 0.0


def check_hallucination(ref: list[str], hyp: list[str]) -> bool:
    """Hallucination heuristic: ref non-empty but hyp >3x longer, min 15 words."""
    return len(ref) > 0 and len(hyp) > max(15, 3 * len(ref))


def compute_aggregate(results: list[dict], model_name: str) -> dict:
    """Match aggregate format from bench_data/results/base_aggregate.json."""
    valid = [r for r in results if r.get("ref_text", "").strip()]
    n = len(valid)

    if n == 0:
        return {"model_name": model_name, "total_utterances": 0}

    hallucination_rate = sum(1 for r in valid if r.get("hallucination", False)) / n
    avg_wer = sum(r["wer"] for r in valid) / n
    avg_cer = sum(r["cer"] for r in valid) / n
    avg_total_ms = sum(r["total_ms"] for r in valid) / n
    avg_rtf = sum(r["rtf"] for r in valid) / n
    deletion_rate = sum(r["deletions"] for r in valid) / n

    by_cat: dict[str, list[dict]] = {}
    for r in valid:
        cat = r.get("category", "unknown")
        by_cat.setdefault(cat, []).append(r)

    by_category = []
    for cat, cat_results in sorted(by_cat.items()):
        c = len(cat_results)
        by_category.append({
            "category": cat,
            "count": c,
            "avg_wer": sum(r["wer"] for r in cat_results) / c,
            "avg_cer": sum(r["cer"] for r in cat_results) / c,
            "avg_rtf": sum(r["rtf"] for r in cat_results) / c,
            "hallucination_rate": sum(1 for r in cat_results if r.get("hallucination", False)) / c,
        })

    return {
        "model_name": model_name,
        "total_utterances": n,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "avg_total_ms": avg_total_ms,
        "avg_rtf": avg_rtf,
        "hallucination_rate": hallucination_rate,
        "deletion_rate": deletion_rate,
        "by_category": by_category,
    }


def run_benchmark():
    jsonl_path = RESULTS_DIR / f"{MODEL_ID}.jsonl"
    aggregate_path = RESULTS_DIR / f"{MODEL_ID}_aggregate.json"

    print(f"\n{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"NeMo: {NEMO_PATH}")
    print(f"{'='*60}")

    import nemo.collections.asr as nemo_asr
    print("Loading model...")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(NEMO_PATH)
    print("Model loaded.")

    with open(CORPUS_PATH) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    results = []

    for idx, entry in enumerate(entries):
        ref_text = entry.get("ref_text", "").strip()
        wav_path = entry["wav_path"]

        if not os.path.exists(wav_path):
            print(f"  [{idx+1}/{len(entries)}] {entry['id']}: SKIP (WAV not found)")
            results.append({
                "id": entry["id"], "wav_path": wav_path, "ref_text": ref_text,
                "hyp_text": "", "category": entry.get("category", "unknown"),
                "wer": 0, "cer": 0, "deletions": 0, "insertions": 0,
                "substitutions": 0, "total_ms": 0, "encoder_ms": 0,
                "decoder_ms": 0, "cross_kv_ms": 0, "audio_secs": 0,
                "rtf": 0, "hallucination": False,
            })
            continue

        audio_secs = get_audio_seconds(wav_path)
        has_ref = bool(ref_text)

        t0 = time.time()
        try:
            output = model.transcribe([wav_path])
            hyp_text = output[0].text.strip() if output else ""
        except Exception as e:
            print(f"  [{idx+1}/{len(entries)}] {entry['id']}: ERROR {e}")
            hyp_text = ""
        total_ms = (time.time() - t0) * 1000

        if has_ref:
            ref_words = normalize(ref_text)
            hyp_words = normalize(hyp_text)
            w, deletions, insertions, substitutions = wer_details(ref_words, hyp_words)
            c = cer(list(ref_text), list(hyp_text))
            hall = check_hallucination(ref_words, hyp_words)
            rtf = (total_ms / 1000) / audio_secs if audio_secs > 0 else 0
        else:
            w, deletions, insertions, substitutions = 0.0, 0.0, 0.0, 0.0
            c = 0.0
            hall = False
            rtf = 0.0

        result = {
            "id": entry["id"], "wav_path": wav_path, "ref_text": ref_text,
            "hyp_text": hyp_text, "category": entry.get("category", "unknown"),
            "wer": w, "cer": c, "deletions": deletions,
            "insertions": insertions, "substitutions": substitutions,
            "total_ms": total_ms, "encoder_ms": 0.0, "decoder_ms": 0.0,
            "cross_kv_ms": 0.0, "audio_secs": audio_secs, "rtf": rtf,
            "hallucination": hall,
        }
        results.append(result)

        status = f"WER={w:.3f} CER={c:.3f} RTF={rtf:.3f}" if has_ref else "skipped"
        print(f"  [{idx+1}/{len(entries)}] {entry['id']}: {total_ms:.0f}ms {status}  \"{hyp_text[:60]}{'...' if len(hyp_text)>60 else ''}\"")

    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nPer-utterance results -> {jsonl_path}")

    aggregate = compute_aggregate(results, MODEL_NAME)
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Aggregate results -> {aggregate_path}")

    eval_results = [r for r in results if r.get("ref_text", "").strip()]
    print(f"\nSummary: {len(eval_results)} evaluable utterances")
    print(f"  avg WER: {aggregate['avg_wer']:.4f}")
    print(f"  avg CER: {aggregate['avg_cer']:.4f}")
    print(f"  avg RTF: {aggregate['avg_rtf']:.4f}")
    print(f"  avg total_ms: {aggregate['avg_total_ms']:.0f}")
    print(f"  hallucination_rate: {aggregate['hallucination_rate']:.4f}")


if __name__ == "__main__":
    run_benchmark()
