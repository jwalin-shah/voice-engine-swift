#!/usr/bin/env python3
"""Benchmark whisper-large-v3-turbo (whisper.cpp) against the voice-engine corpus.

Outputs:
  bench_data/results/whisper-large-v3-turbo.jsonl        -- per-utterance results
  bench_data/results/whisper-large-v3-turbo_aggregate.json  -- summary stats
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# -- Configuration -----------------------------------------------------------

WHISPER_CLI = os.environ.get(
    "WHISPER_CLI",
    "/tmp/whisper.cpp/build/bin/whisper-cli",
)
MODEL_PATH = os.environ.get(
    "WHISPER_MODEL",
    "/tmp/whisper.cpp/models/ggml-large-v3-turbo.bin",
)
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "bench_data", "corpus.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "bench_data", "results")
OUT_JSONL = os.path.join(OUT_DIR, "whisper-large-v3-turbo.jsonl")
OUT_AGG = os.path.join(OUT_DIR, "whisper-large-v3-turbo_aggregate.json")

# -- WER helpers -------------------------------------------------------------

def normalize(text):
    upper = text.upper()
    cleaned = re.sub(r"[^A-Z' ]", " ", upper)
    return [w for w in cleaned.split() if w]


def wer(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
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
    return dp[n][m] / n


def cer(ref, hyp):
    ref_chars = list(" ".join(ref))
    hyp_chars = list(" ".join(hyp))
    n, m = len(ref_chars), len(hyp_chars)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n


# -- Whisper CLI runner ------------------------------------------------------

def transcribe(wav_path):
    args = [
        WHISPER_CLI,
        "-m", MODEL_PATH,
        "-f", wav_path,
        "-l", "en",
        "--no-timestamps",
    ]

    t_start = time.perf_counter()
    proc = subprocess.run(args, capture_output=True, text=True, timeout=300)
    t_end = time.perf_counter()
    wall_ms = (t_end - t_start) * 1000.0

    hyp_text = proc.stdout.strip()

    timing_ms = wall_ms
    for line in proc.stderr.splitlines():
        m = re.search(r"total time =\s+([\d.]+)\s+ms", line)
        if m:
            timing_ms = float(m.group(1))
            break

    return hyp_text, timing_ms


# -- Main --------------------------------------------------------------------

def load_corpus():
    entries = []
    with open(CORPUS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("ref_text"):
                entries.append(entry)
    return entries


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    entries = load_corpus()
    print(f"Loaded {len(entries)} evaluable entries from corpus")

    done_ids = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])

    results_fh = open(OUT_JSONL, "w")  # overwrite for F16 re-run
    results = []

    for i, entry in enumerate(entries):
        eid = entry["id"]
        wav_path = entry["wav_path"]
        ref_text = entry["ref_text"]
        category = entry.get("category", "unknown")
        audio_secs = entry.get("audio_secs")

        if not os.path.exists(wav_path):
            print(f"[{i+1}/{len(entries)}] SKIP {eid} (wav missing)")
            continue

        print(f"[{i+1}/{len(entries)}] {eid} ({category})", end=" ", flush=True)

        try:
            hyp_text, total_ms = transcribe(wav_path)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            result = {"id": eid, "wav_path": wav_path, "ref_text": ref_text, "hyp_text": "",
                       "category": category, "wer": float("nan"), "cer": float("nan"),
                       "rtf": float("nan"), "hallucination": False, "audio_secs": audio_secs,
                       "total_ms": 300_000.0, "error": "timeout"}
            json.dump(result, results_fh); results_fh.write("\n"); results_fh.flush()
            results.append(result)
            continue
        except Exception as e:
            print(f"ERROR: {e}")
            result = {"id": eid, "wav_path": wav_path, "ref_text": ref_text, "hyp_text": "",
                       "category": category, "wer": float("nan"), "cer": float("nan"),
                       "rtf": float("nan"), "hallucination": False, "audio_secs": audio_secs,
                       "total_ms": 0.0, "error": str(e)}
            json.dump(result, results_fh); results_fh.write("\n"); results_fh.flush()
            results.append(result)
            continue

        ref_words = normalize(ref_text)
        hyp_words = normalize(hyp_text)

        w = wer(ref_words, hyp_words)
        c = cer(ref_words, hyp_words)

        if audio_secs and audio_secs > 0:
            rtf = (total_ms / 1000.0) / audio_secs
        else:
            rtf = 0.0

        hallucination = w > 0.5 or (len(hyp_words) > len(ref_words) * 3 and len(ref_words) > 3)

        result = {"id": eid, "wav_path": wav_path, "ref_text": ref_text, "hyp_text": hyp_text,
                   "category": category, "wer": w, "cer": c, "rtf": rtf,
                   "hallucination": hallucination, "audio_secs": audio_secs, "total_ms": total_ms}
        json.dump(result, results_fh); results_fh.write("\n"); results_fh.flush()
        results.append(result)

        print(f"WER={w:.3f} CER={c:.3f} RTF={rtf:.3f} ({total_ms:.0f}ms)")

    results_fh.close()

    # -- Aggregate -----------------------------------------------------------
    all_results = []
    with open(OUT_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                all_results.append(json.loads(line))

    valid = [r for r in all_results if "error" not in r]

    if not valid:
        print("\nNo valid results -- cannot compute aggregate.")
        return

    avg_wer = sum(r["wer"] for r in valid) / len(valid)
    avg_cer = sum(r["cer"] for r in valid) / len(valid)
    avg_rtf = sum(r["rtf"] for r in valid) / len(valid)
    avg_total_ms = sum(r["total_ms"] for r in valid) / len(valid)
    hallucination_count = sum(1 for r in valid if r.get("hallucination"))
    hallucination_rate = hallucination_count / len(valid)

    cats = {}
    for r in valid:
        cat = r.get("category", "unknown")
        cats.setdefault(cat, []).append(r)

    by_category = []
    for cat, cat_results in sorted(cats.items()):
        by_category.append({
            "category": cat,
            "count": len(cat_results),
            "avg_wer": sum(r["wer"] for r in cat_results) / len(cat_results),
            "avg_cer": sum(r["cer"] for r in cat_results) / len(cat_results),
            "avg_rtf": sum(r["rtf"] for r in cat_results) / len(cat_results),
            "hallucination_rate": sum(1 for r in cat_results if r.get("hallucination")) / len(cat_results),
        })

    aggregate = {
        "model_name": "whisper-large-v3-turbo",
        "total_utterances": len(valid),
        "total_with_errors": len(all_results) - len(valid),
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "avg_rtf": avg_rtf,
        "avg_total_ms": avg_total_ms,
        "hallucination_rate": hallucination_rate,
        "by_category": by_category,
    }

    with open(OUT_AGG, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n=== AGGREGATE ===")
    print(f"Utterances: {len(valid)} valid / {len(all_results)} total")
    print(f"Avg WER:    {avg_wer:.4f}")
    print(f"Avg CER:    {avg_cer:.4f}")
    print(f"Avg RTF:    {avg_rtf:.4f}")
    print(f"Avg total:  {avg_total_ms:.0f} ms")
    print(f"Hallucination rate: {hallucination_rate:.4f}")
    print(f"\nBy category:")
    for bc in by_category:
        print(f"  {bc['category']:20s} n={bc['count']:3d}  WER={bc['avg_wer']:.4f}  CER={bc['avg_cer']:.4f}  RTF={bc['avg_rtf']:.4f}")
    print(f"\nWrote: {OUT_JSONL}")
    print(f"Wrote: {OUT_AGG}")


if __name__ == "__main__":
    main()
