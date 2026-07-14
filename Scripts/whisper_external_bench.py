#!/usr/bin/env python3
"""whisper.cpp benchmark on external corpus — Metal GPU, real ground truth."""
import json, re, subprocess, time, os
from pathlib import Path
from collections import defaultdict

WHISPER = "/tmp/whisper.cpp/build/bin/whisper-cli"
MODEL = "/tmp/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin"
CORPUS = Path("bench_data/corpus_external.jsonl")
OUT_JSONL = Path("bench_data/results/whisper-large-v3-turbo-q5_0-external.jsonl")
OUT_AGG = Path("bench_data/results/whisper-large-v3-turbo-q5_0-external_aggregate.json")

def norm(t): return [w for w in re.sub(r"[^A-Z' ]", " ", t.upper()).split() if w]

def calc_wer(ref_w, hyp_w):
    n, m = len(ref_w), len(hyp_w)
    if n == 0: return 0.0 if m == 0 else 1.0
    if m == 0: return 1.0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = dp[i-1][j-1] if ref_w[i-1]==hyp_w[j-1] else 1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
    return dp[n][m]/n

with open(CORPUS) as f:
    entries = [json.loads(l) for l in f if l.strip()]
print(f"Corpus: {len(entries)} clips")

results = []
for i, e in enumerate(entries):
    t0 = time.time()
    proc = subprocess.run(
        [WHISPER, "-m", MODEL, "-f", e["wav_path"], "--no-timestamps", "-l", "en", "-t", "4"],
        capture_output=True, text=True, timeout=120
    )
    ms = (time.time() - t0) * 1000

    # Parse whisper output — transcript on timestamped lines
    output = proc.stderr + "\n" + proc.stdout
    text_parts = []
    for line in output.split("\n"):
        line = line.strip()
        # Match: [00:00:00.000 --> 00:00:08.740]   Text here
        if line.startswith("[00:") and "-->" in line:
            # Extract text after the timestamp bracket
            idx = line.index("]  ") + 3 if "]  " in line else len(line)
            text = line[idx:].strip()
            if text and not text.startswith("whisper_"):
                text_parts.append(text)
    hyp = " ".join(text_parts).strip()
    hyp = re.sub(r'\[BLANK_AUDIO\]', '', hyp)
    hyp = ' '.join(hyp.split())

    w = calc_wer(norm(e["ref_text"]), norm(hyp))
    r = {
        "id": e["id"], "wav_path": e["wav_path"], "ref_text": e["ref_text"],
        "hyp_text": hyp, "category": e["category"],
        "wer": w, "cer": 0, "total_ms": ms,
        "audio_secs": e.get("audio_secs", 0),
        "rtf": (ms/1000) / max(e.get("audio_secs", 1), 0.001),
        "hallucination": w > 0.8 or (len(norm(hyp)) > 3 * max(len(norm(e["ref_text"])), 1))
    }
    results.append(r)

    if (i+1) % 20 == 0 or i == 0:
        with open(OUT_JSONL, "w") as f:
            for r2 in results: f.write(json.dumps(r2) + "\n")
        avg = sum(x["wer"] for x in results) / len(results)
        print(f"  [{i+1}/{len(entries)}] running WER={avg:.4f}  last: \"{hyp[:60]}\"")

# Final write
with open(OUT_JSONL, "w") as f:
    for r in results: f.write(json.dumps(r) + "\n")

# Aggregate
by_cat = defaultdict(lambda: {"wers": [], "ms": [], "rtfs": []})
for r in results:
    by_cat[r["category"]]["wers"].append(r["wer"])
    by_cat[r["category"]]["ms"].append(r["total_ms"])
    by_cat[r["category"]]["rtfs"].append(r["rtf"])

avg_w = sum(r["wer"] for r in results) / len(results)
avg_ms = sum(r["total_ms"] for r in results) / len(results)
avg_rtf = sum(r["rtf"] for r in results) / len(results)
halls = sum(1 for r in results if r["hallucination"]) / len(results)

agg = {
    "model_name": "whisper-large-v3-turbo-q5_0 (Metal GPU)",
    "total_utterances": len(results),
    "avg_wer": avg_w, "avg_total_ms": avg_ms, "avg_rtf": avg_rtf,
    "hallucination_rate": halls,
    "by_category": [
        {"category": cat, "count": len(d["wers"]),
         "avg_wer": sum(d["wers"])/len(d["wers"]),
         "avg_total_ms": sum(d["ms"])/len(d["ms"]),
         "hallucination_rate": sum(1 for w in d["wers"] if w > 0.8)/len(d["wers"])}
        for cat, d in sorted(by_cat.items())
    ]
}
with open(OUT_AGG, "w") as f:
    json.dump(agg, f, indent=2)

print(f"\n=== Whisper Large v3 Turbo q5_0 (Metal GPU) ===")
print(f"{len(results)} utterances, WER={avg_w:.4f}, {avg_ms:.0f}ms, RTF={avg_rtf:.3f}, hall={halls:.1%}")
for cat in sorted(by_cat.keys()):
    d = by_cat[cat]; w = sum(d["wers"])/len(d["wers"]); m = sum(d["ms"])/len(d["ms"])
    print(f"  {cat:<30s} WER={w:.4f}  {m:.0f}ms  n={len(d['wers'])}")
