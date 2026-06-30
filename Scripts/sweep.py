#!/usr/bin/env python3
"""
Run the Moonshine CoreML pipeline against a stratified sample of LibriSpeech
clips. Report per-stage latency by audio length bucket + WER vs ground truth.
"""
import warnings, contextlib, io, json, time, re
warnings.filterwarnings("ignore")

import sys
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict
import coremltools as ct
from transformers import AutoTokenizer

MODEL_DIR = Path.home() / ".cache" / "moonshine-coreml" / "tiny-streaming"
LIBRI_ROOT = Path("/tmp/librispeech/LibriSpeech/test-clean")
SAMPLE_RATE = 16000
NL, H, D = 6, 8, 36
S_MAX, S_ENC_MAX, ROT_DIM = 128, 500, 32
ENC_WINDOW = 160000  # 10s pad


def normalize(text: str) -> list:
    """LibriSpeech-style normalization: uppercase, strip punctuation, split."""
    text = text.upper()
    text = re.sub(r"[^A-Z' ]", " ", text)
    return [w for w in text.split() if w]


def wer(ref: list, hyp: list) -> float:
    """Word error rate via Levenshtein on word lists."""
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
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m] / n


def load_transcripts():
    """Map utterance_id → ground truth text."""
    out = {}
    for trans_file in LIBRI_ROOT.rglob("*.trans.txt"):
        for line in trans_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, text = line.split(" ", 1)
            out[utt_id] = text
    return out


def pick_clips(transcripts, n_per_bucket=5):
    """Pick stratified clips across duration buckets."""
    buckets = {
        "0.5-1.5s": [],
        "1.5-3s":   [],
        "3-5s":     [],
        "5-8s":     [],
        "8-15s":    [],
    }
    flacs = list(LIBRI_ROOT.rglob("*.flac"))
    # Use first N from each bucket — deterministic
    for flac in sorted(flacs):
        utt_id = flac.stem
        if utt_id not in transcripts:
            continue
        info = sf.info(str(flac))
        dur = info.frames / info.samplerate
        for name, (lo, hi) in [
            ("0.5-1.5s", (0.5, 1.5)),
            ("1.5-3s",   (1.5, 3.0)),
            ("3-5s",     (3.0, 5.0)),
            ("5-8s",     (5.0, 8.0)),
            ("8-15s",    (8.0, 15.0)),
        ]:
            if lo <= dur < hi and len(buckets[name]) < n_per_bucket:
                buckets[name].append((flac, dur, transcripts[utt_id]))
    return buckets


class MoonshinePipeline:
    def __init__(self):
        print("Loading models…")
        self.encoder = ct.models.MLModel(str(MODEL_DIR / "encoder.mlpackage"))
        self.decoder = ct.models.MLModel(str(MODEL_DIR / "decoder_stateful.mlpackage"),
                                         compute_units=ct.ComputeUnit.CPU_AND_GPU)
        self.tok = AutoTokenizer.from_pretrained("UsefulSensors/moonshine-tiny")

        w = np.load(str(MODEL_DIR / "cross_kv_weights.npz"))
        self.kw = [w[f"layer{i}_k_weight"] for i in range(NL)]
        self.vw = [w[f"layer{i}_v_weight"] for i in range(NL)]
        self.kb = [w.get(f"layer{i}_k_bias") for i in range(NL)]
        self.vb = [w.get(f"layer{i}_v_bias") for i in range(NL)]
        cos_t = w["cos_tables"]; sin_t = w["sin_tables"]
        self.cos_tables = [cos_t[i].reshape(1, 1, 1, -1).astype(np.float32) for i in range(S_MAX)]
        self.sin_tables = [sin_t[i].reshape(1, 1, 1, -1).astype(np.float32) for i in range(S_MAX)]

        self.enc_in = list(self.encoder.input_description)[0]
        self.enc_out = list(self.encoder.output_description)[0]

        # Warmup with real-looking noise (avoid the all-zero overflow warnings).
        warmup = np.random.randn(1, ENC_WINDOW).astype(np.float32) * 0.05
        _ = self.encoder.predict({self.enc_in: warmup})
        print("  Ready.")

    def transcribe(self, audio: np.ndarray):
        times = {}

        # Pad to 10s window.
        t0 = time.perf_counter()
        if audio.ndim == 1:
            audio = audio[None, :]
        n = audio.shape[1]
        if n < ENC_WINDOW:
            audio = np.pad(audio, ((0,0),(0,ENC_WINDOW - n)))
        elif n > ENC_WINDOW:
            audio = audio[:, :ENC_WINDOW]
        times["preprocess_ms"] = (time.perf_counter() - t0) * 1000

        # Encoder.
        t0 = time.perf_counter()
        hidden = self.encoder.predict({self.enc_in: audio.astype(np.float32)})[self.enc_out]
        if hidden.ndim == 2:
            hidden = hidden[None]
        S_enc = hidden.shape[1]
        times["encoder_ms"] = (time.perf_counter() - t0) * 1000

        # Cross-KV projection.
        t0 = time.perf_counter()
        ck_list, cv_list = [], []
        for i in range(NL):
            k = hidden @ self.kw[i].T
            v = hidden @ self.vw[i].T
            if self.kb[i] is not None: k = k + self.kb[i]
            if self.vb[i] is not None: v = v + self.vb[i]
            k = k.reshape(1, S_enc, H, D).transpose(0, 2, 1, 3)
            v = v.reshape(1, S_enc, H, D).transpose(0, 2, 1, 3)
            ck_list.append(k); cv_list.append(v)
        cross_k = np.stack(ck_list).astype(np.float32)
        cross_v = np.stack(cv_list).astype(np.float32)
        pad_amt = S_ENC_MAX - S_enc
        if pad_amt > 0:
            cross_k = np.pad(cross_k, ((0,0),(0,0),(0,0),(0,pad_amt),(0,0))).astype(np.float32)
            cross_v = np.pad(cross_v, ((0,0),(0,0),(0,0),(0,pad_amt),(0,0))).astype(np.float32)
        cross_mask = np.full((1,1,1,S_ENC_MAX), -1e4, dtype=np.float32)
        cross_mask[..., :S_enc] = 0.0
        times["kv_projection_ms"] = (time.perf_counter() - t0) * 1000

        # Decoder loop.
        t0 = time.perf_counter()
        state = self.decoder.make_state()
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            state.write_state("cross_k", cross_k)
            state.write_state("cross_v", cross_v)
            state.write_state("cross_mask", cross_mask)

        BOS, EOS = 1, self.tok.eos_token_id or 2
        attn_mask = np.full((1,1,1,S_MAX), -1e4, dtype=np.float32)
        attn_mask[..., 0] = 0.0
        onehot = np.zeros((1,1,S_MAX,1), dtype=np.float32)
        onehot[0,0,0,0] = 1.0
        tokens = [BOS]
        for step in range(S_MAX - 1):
            out = self.decoder.predict({
                "input_ids":    np.array([[tokens[-1]]], dtype=np.int32),
                "attn_mask":    attn_mask,
                "cos":          self.cos_tables[step],
                "sin":          self.sin_tables[step],
                "write_onehot": onehot,
            }, state=state)
            nxt = int(np.asarray(out["logits"])[0, 0].argmax())
            tokens.append(nxt)
            if nxt == EOS:
                break
            nxt_pos = step + 1
            if nxt_pos < S_MAX:
                attn_mask[..., nxt_pos] = 0.0
                onehot = np.zeros((1,1,S_MAX,1), dtype=np.float32)
                onehot[0,0,nxt_pos,0] = 1.0
        times["decoder_ms"] = (time.perf_counter() - t0) * 1000
        times["n_tokens"] = len(tokens) - 1  # exclude BOS

        # Detokenize.
        t0 = time.perf_counter()
        text = self.tok.decode(tokens, skip_special_tokens=True).strip()
        times["detokenize_ms"] = (time.perf_counter() - t0) * 1000
        times["total_ms"] = (
            times["preprocess_ms"] + times["encoder_ms"]
            + times["kv_projection_ms"] + times["decoder_ms"] + times["detokenize_ms"]
        )

        return text, times


def main():
    transcripts = load_transcripts()
    print(f"Loaded {len(transcripts)} ground-truth transcripts")

    buckets = pick_clips(transcripts, n_per_bucket=5)
    total = sum(len(v) for v in buckets.values())
    print(f"Picked {total} clips across {len(buckets)} duration buckets\n")

    pipe = MoonshinePipeline()
    pipe.transcribe(np.zeros((1, 16000), dtype=np.float32))  # extra warmup

    results = []
    per_bucket = defaultdict(list)
    for bucket_name, clips in buckets.items():
        print(f"\n── {bucket_name} ──────────────────────────────────────")
        for flac, dur, ground_truth in clips:
            audio, sr = sf.read(str(flac), dtype="float32")
            if sr != SAMPLE_RATE:
                continue
            hyp_text, times = pipe.transcribe(audio)
            ref_words = normalize(ground_truth)
            hyp_words = normalize(hyp_text)
            er = wer(ref_words, hyp_words)
            record = {
                "id": flac.stem, "duration_s": round(dur, 2),
                "ref": ground_truth, "hyp": hyp_text,
                "wer": round(er, 3), **{k: round(v, 2) if isinstance(v, float) else v
                                        for k, v in times.items()},
            }
            results.append(record)
            per_bucket[bucket_name].append(record)
            rtf = times["total_ms"] / (dur * 1000)
            print(f"  {flac.stem}  {dur:>4.1f}s → {times['total_ms']:>5.1f}ms "
                  f"(RTF={rtf:>5.3f}, {times['n_tokens']:>2}t, WER={er:.2f})")
            print(f"     hyp: \"{hyp_text[:90]}\"")

    print("\n\n══ Per-bucket summary ══════════════════════════════════════════════════")
    print(f"  {'bucket':>10} {'n':>3} {'enc':>6} {'kvp':>5} {'dec':>6} {'total':>7} "
          f"{'tok':>4} {'RTF':>6} {'WER':>5}")
    for name in ["0.5-1.5s", "1.5-3s", "3-5s", "5-8s", "8-15s"]:
        rs = per_bucket[name]
        if not rs:
            continue
        n = len(rs)
        enc = np.mean([r["encoder_ms"] for r in rs])
        kvp = np.mean([r["kv_projection_ms"] for r in rs])
        dec = np.mean([r["decoder_ms"] for r in rs])
        tot = np.mean([r["total_ms"] for r in rs])
        tok = np.mean([r["n_tokens"] for r in rs])
        dur = np.mean([r["duration_s"] for r in rs])
        rtf = tot / (dur * 1000)
        avg_wer = np.mean([r["wer"] for r in rs])
        print(f"  {name:>10} {n:>3} {enc:>5.1f}  {kvp:>4.1f}  {dec:>5.1f}  {tot:>6.1f}  "
              f"{tok:>3.0f}  {rtf:>5.3f}  {avg_wer:>4.2f}")

    print("\nOverall:")
    all_total = np.mean([r["total_ms"] for r in results])
    all_wer = np.mean([r["wer"] for r in results])
    all_dur = np.mean([r["duration_s"] for r in results])
    print(f"  Mean E2E latency: {all_total:.1f} ms over {len(results)} clips")
    print(f"  Mean RTF: {all_total / (all_dur * 1000):.3f}  ({1000 / all_total:.0f} utterances/sec)")
    print(f"  Mean WER: {all_wer:.2%}")

    Path("/tmp/sweep_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nFull results → /tmp/sweep_results.json")


if __name__ == "__main__":
    main()
