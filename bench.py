#!/usr/bin/env python3
"""
Benchmark the Moonshine CoreML pipeline with per-stage timing.

Measures:
  1. Audio load + pad/clip
  2. Encoder forward pass (ANE)
  3. Cross-KV projection (CPU)
  4. Decoder loop (CPU, stateful) — per-token breakdown
  5. Token decode → text
  6. End-to-end wall clock

Usage:
  python3 bench.py                    # Use synthetic silence
  python3 bench.py test.wav           # Use a real WAV
  python3 bench.py --iterations 10    # Average over N runs
"""

import math, time, sys, os, json, struct, tempfile
import numpy as np
from pathlib import Path

SAMPLE_RATE = 16000
ENC_WINDOW = 10 * SAMPLE_RATE
OVERLAP_SAMPLES = 2 * SAMPLE_RATE
MIN_NEW_AUDIO_SAMPLES = SAMPLE_RATE
MODEL_DIR = Path.home() / ".cache" / "moonshine-coreml" / "tiny-streaming"
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
WAV_SUBFORMAT_GUID_TAIL = b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


def read_wav_chunks(path: str):
    """Read basic RIFF/WAV chunks needed for PCM and IEEE-float WAV input."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"RIFF":
                raise ValueError("missing RIFF header")
            riff_size_raw = f.read(4)
            if len(riff_size_raw) < 4:
                raise ValueError("truncated RIFF header")
            riff_size = struct.unpack("<I", riff_size_raw)[0]
            if riff_size < 4:
                raise ValueError(f"invalid RIFF size: {riff_size}")
            file_size = f.seek(0, os.SEEK_END)
            riff_end = 8 + riff_size
            if riff_end > file_size:
                raise ValueError(
                    f"truncated RIFF data: header declares {riff_size} bytes, "
                    f"file has {file_size - 8} bytes"
                )
            f.seek(8)
            if f.read(4) != b"WAVE":
                raise ValueError("missing WAVE header")

            fmt = None
            data = None
            while f.tell() < riff_end:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    raise ValueError("truncated chunk header")
                chunk_size_raw = f.read(4)
                if len(chunk_size_raw) < 4:
                    raise ValueError("truncated chunk size")
                chunk_size = struct.unpack("<I", chunk_size_raw)[0]
                chunk_end = f.tell() + chunk_size
                if chunk_end > riff_end:
                    chunk_name = chunk_id.decode("ascii", errors="replace")
                    raise ValueError(
                        f"chunk {chunk_name} extends beyond declared RIFF data"
                    )
                chunk_data = f.read(chunk_size)
                if len(chunk_data) < chunk_size:
                    raise ValueError(
                        f"truncated chunk {chunk_id.decode('ascii', errors='replace')}: "
                        f"expected {chunk_size} bytes, read {len(chunk_data)}"
                    )
                if chunk_id == b"fmt ":
                    fmt = chunk_data
                elif chunk_id == b"data":
                    data = chunk_data
                if chunk_size % 2:
                    if f.tell() >= riff_end:
                        raise ValueError(
                            f"missing pad byte after odd-sized chunk "
                            f"{chunk_id.decode('ascii', errors='replace')}"
                        )
                    f.read(1)
    except OSError as exc:
        raise ValueError(str(exc)) from exc

    if fmt is None:
        raise ValueError("missing fmt chunk")
    if data is None:
        raise ValueError("missing data chunk")
    if len(fmt) < 16:
        raise ValueError("truncated fmt chunk")

    container_audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack("<HHIIHH", fmt[:16])
    audio_format = container_audio_format
    if container_audio_format == WAVE_FORMAT_EXTENSIBLE:
        if len(fmt) < 40:
            raise ValueError("truncated WAVE_FORMAT_EXTENSIBLE fmt chunk")
        extension_size = struct.unpack("<H", fmt[16:18])[0]
        if extension_size < 22:
            raise ValueError(f"invalid WAVE_FORMAT_EXTENSIBLE extension size: {extension_size}")
        available_extension_size = len(fmt) - 18
        if extension_size > available_extension_size:
            raise ValueError(
                f"truncated WAVE_FORMAT_EXTENSIBLE extension: declared {extension_size} bytes, "
                f"only {available_extension_size} available"
            )
        valid_bits_per_sample = struct.unpack("<H", fmt[18:20])[0]
        if valid_bits_per_sample == 0 or valid_bits_per_sample > bits_per_sample:
            raise ValueError(
                f"invalid WAVE_FORMAT_EXTENSIBLE valid bits per sample: "
                f"{valid_bits_per_sample} for {bits_per_sample}-bit samples"
            )
        subformat = fmt[24:40]
        if subformat[2:] != WAV_SUBFORMAT_GUID_TAIL:
            raise ValueError("unsupported WAVE_FORMAT_EXTENSIBLE subformat GUID")
        audio_format = struct.unpack("<H", subformat[:2])[0]

    if bits_per_sample % 8 != 0:
        raise ValueError(f"unsupported non-byte-aligned sample width: {bits_per_sample} bits")
    sample_width = bits_per_sample // 8
    expected_block_align = channels * sample_width
    if expected_block_align <= 0:
        raise ValueError("invalid WAV channel count or sample width")
    if block_align != expected_block_align:
        raise ValueError(
            f"invalid block alignment: expected {expected_block_align}, got {block_align}"
        )
    expected_byte_rate = sample_rate * expected_block_align
    if byte_rate != expected_byte_rate:
        raise ValueError(
            f"invalid byte rate: expected {expected_byte_rate}, got {byte_rate}"
        )
    if len(data) % expected_block_align != 0:
        raise ValueError(
            f"truncated sample data: {len(data)} bytes is not divisible by frame size {expected_block_align}"
        )
    frames = len(data) // expected_block_align
    return {
        "audio_format": audio_format,
        "container_audio_format": container_audio_format,
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "frames": frames,
        "raw": data,
    }


def wav_audio_format(path: str):
    """Return the WAV fmt audio format code, e.g. 1=PCM or 3=IEEE float."""
    return read_wav_chunks(path)["audio_format"]


def validated_audio_wav_info(path: str):
    """Validate WAV metadata and return parsed WAV info."""
    wav_path = Path(path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    if not wav_path.is_file():
        raise ValueError(f"Audio path is not a file: {wav_path}")

    try:
        info = read_wav_chunks(str(wav_path))
    except ValueError as exc:
        reason = str(exc) or exc.__class__.__name__
        raise ValueError(f"Audio file is not a readable WAV: {wav_path} ({reason})") from exc
    channels = info["channels"]
    sample_width = info["sample_width"]
    sample_rate = info["sample_rate"]
    frames = info["frames"]
    raw = info["raw"]
    audio_format = info["audio_format"]

    if channels != 1:
        raise ValueError(f"Expected mono WAV, got {channels} channels: {wav_path}")
    if audio_format == 1 and sample_width == 2:
        pass
    elif audio_format == 3 and sample_width == 4:
        pass
    else:
        raise ValueError(
            f"Expected 16-bit PCM or 32-bit float WAV, got format={audio_format} "
            f"with {sample_width * 8}-bit samples: {wav_path}"
        )
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz WAV, got {sample_rate} Hz: {wav_path}")
    if frames <= 0:
        raise ValueError(f"Audio file has no samples: {wav_path}")
    expected_bytes = frames * channels * sample_width
    if len(raw) != expected_bytes:
        raise ValueError(
            f"Audio file is truncated: expected {expected_bytes} bytes of sample data, "
            f"read {len(raw)} bytes: {wav_path}"
        )
    if audio_format == 3 and sample_width == 4:
        samples = np.frombuffer(raw, dtype="<f4")
        if not np.all(np.isfinite(samples)):
            raise ValueError(f"Audio file contains non-finite float samples: {wav_path}")

    return info


def inspect_audio_wav(path: str) -> tuple[int, float]:
    """Validate WAV metadata and return (frame_count, duration_s)."""
    info = validated_audio_wav_info(path)
    frames = info["frames"]
    return frames, frames / SAMPLE_RATE


def load_audio_wav(path: str) -> np.ndarray:
    """Load mono WAV as float32 [-1, 1]. Supports 16-bit PCM and 32-bit float."""
    info = validated_audio_wav_info(path)
    raw = info["raw"]
    if info["audio_format"] == 1 and info["sample_width"] == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if info["audio_format"] == 3 and info["sample_width"] == 4:
        samples = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        if not np.all(np.isfinite(samples)):
            raise ValueError(f"Audio file contains non-finite float samples: {path}")
        return samples
    raise ValueError(f"Unsupported WAV format after validation: {path}")


def generate_test_audio(duration_s: float = 3.0) -> np.ndarray:
    """Generate synthetic audio with a 440 Hz tone + noise."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.05 * np.random.randn(n).astype(np.float32)
    audio = tone + noise
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.95
    return audio


def dedup_overlap(prev_text: str, new_text: str) -> str:
    """Drop repeated leading text caused by overlapping audio chunks."""
    import re

    prev_sents = re.split(r'(?<=[.!?])\s+', prev_text.strip())
    new_sents = re.split(r'(?<=[.!?])\s+', new_text.strip())
    if not prev_sents or not new_sents:
        return new_text

    def norm(sentence: str) -> str:
        return sentence.strip().lower().rstrip(".,!?;")

    tail = [norm(s) for s in (prev_sents[-2:] if len(prev_sents) >= 2 else prev_sents[-1:])]
    for skip in range(min(len(new_sents), 4)):
        head = norm(new_sents[skip])
        for prev in tail:
            if not head or not prev:
                continue
            if (
                len(prev) > 8
                and len(head) > 8
                and (
                    head.startswith(prev)
                    or prev.startswith(head)
                    or prev[:15] == head[:15]
                    or prev in head
                    or head in prev
                )
            ):
                return " ".join(new_sents[skip + 1:]).strip()
        if len(new_sents[skip].split()) <= 4:
            continue
        break

    return new_text


class MoonshineBench:
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.consts = {}
        self.kw = self.vw = self.kb = self.vb = None
        self.cos_tables = self.sin_tables = None
        self.tokenizer = None
        self.enc_in = self.enc_out = None
        self.ready = False

    def load(self):
        import coremltools as ct
        from transformers import AutoTokenizer

        enc_path = MODEL_DIR / "encoder.mlpackage"
        dec_path = MODEL_DIR / "decoder_stateful.mlpackage"
        w_path = MODEL_DIR / "cross_kv_weights.npz"

        if not (enc_path.exists() and dec_path.exists() and w_path.exists()):
            print(f"ERROR: Models not found in {MODEL_DIR}")
            print("Run: ./build.sh setup")
            return False

        t0 = time.perf_counter()
        self.encoder = ct.models.MLModel(str(enc_path))
        t1 = time.perf_counter()
        self.decoder = ct.models.MLModel(str(dec_path), compute_units=ct.ComputeUnit.CPU_ONLY)
        t2 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained("UsefulSensors/moonshine-tiny")
        t3 = time.perf_counter()

        w = np.load(str(w_path))
        NL = int(w["NL"]); H = int(w["H"]); D = int(w["D"])
        HID = int(w["HID"]); S_MAX = int(w["S_MAX"])
        self.consts = {"NL": NL, "H": H, "D": D, "HID": HID, "S_MAX": S_MAX}
        self.kw = [w[f"layer{i}_k_weight"] for i in range(NL)]
        self.vw = [w[f"layer{i}_v_weight"] for i in range(NL)]
        self.kb = [w.get(f"layer{i}_k_bias") for i in range(NL)]
        self.vb = [w.get(f"layer{i}_v_bias") for i in range(NL)]

        cos_t = w["cos_tables"]
        sin_t = w["sin_tables"]
        self.cos_tables = [cos_t[i].reshape(1, 1, 1, -1) for i in range(S_MAX)]
        self.sin_tables = [sin_t[i].reshape(1, 1, 1, -1) for i in range(S_MAX)]

        self.enc_in = list(self.encoder.input_description)[0]
        self.enc_out = list(self.encoder.output_description)[0]
        self.ready = True

        print(f"  encoder load: {(t1-t0)*1000:.0f}ms")
        print(f"  decoder load: {(t2-t1)*1000:.0f}ms")
        print(f"  tokenizer load: {(t3-t2)*1000:.0f}ms")
        print(f"  total load: {(t3-t0)*1000:.0f}ms")
        return True

    def transcribe(self, audio: np.ndarray):
        """Full pipeline with per-stage timing."""
        NL = self.consts["NL"]; H = self.consts["H"]; D = self.consts["D"]
        S_MAX = self.consts["S_MAX"]

        times = {}

        # 1. Pad to encoder window. Call transcribe_chunked for longer audio.
        t0 = time.perf_counter()
        audio = audio.astype(np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        n = audio.shape[1]
        if n < ENC_WINDOW:
            audio = np.pad(audio, ((0, 0), (0, ENC_WINDOW - n)))
        elif n > ENC_WINDOW:
            raise ValueError(
                f"Audio has {n} samples; use transcribe_chunked for inputs longer than {ENC_WINDOW}"
            )
        times["preprocess"] = time.perf_counter() - t0

        # 2. Encoder forward.
        t0 = time.perf_counter()
        hidden = self.encoder.predict({self.enc_in: audio})[self.enc_out]
        if hidden.ndim == 2:
            hidden = hidden[None]
        S_enc = hidden.shape[1]
        S_ENC_MAX = 500
        if S_enc > S_ENC_MAX:
            raise ValueError(
                f"Encoder output has {S_enc} frames, exceeding decoder limit {S_ENC_MAX}"
            )
        times["encoder"] = time.perf_counter() - t0

        # 3. Cross-KV projection.
        t0 = time.perf_counter()
        cross_k_list, cross_v_list = [], []
        for i in range(NL):
            k = hidden @ self.kw[i].T
            v = hidden @ self.vw[i].T
            if self.kb[i] is not None:
                k = k + self.kb[i]
            if self.vb[i] is not None:
                v = v + self.vb[i]
            k = k.reshape(1, S_enc, H, D).transpose(0, 2, 1, 3)
            v = v.reshape(1, S_enc, H, D).transpose(0, 2, 1, 3)
            cross_k_list.append(k)
            cross_v_list.append(v)
        cross_k = np.stack(cross_k_list).astype(np.float32)
        cross_v = np.stack(cross_v_list).astype(np.float32)

        # Pad cross_k/v to S_ENC_MAX=500 (CoreML decoder input shape),
        # and build a cross_mask that exposes only the actual S_enc frames.
        pad_amt = S_ENC_MAX - S_enc
        if pad_amt > 0:
            cross_k = np.pad(cross_k, ((0,0),(0,0),(0,0),(0,pad_amt),(0,0))).astype(np.float32)
            cross_v = np.pad(cross_v, ((0,0),(0,0),(0,0),(0,pad_amt),(0,0))).astype(np.float32)
        cross_mask = np.full((1, 1, 1, S_ENC_MAX), -1e4, dtype=np.float32)
        cross_mask[..., :S_enc] = 0.0
        times["kv_projection"] = time.perf_counter() - t0

        # 4. Decoder loop.
        t0 = time.perf_counter()
        state = self.decoder.make_state()

        # Set cross-attention state tensors via state API (coremltools 9).
        state.write_state("cross_k", cross_k)
        state.write_state("cross_v", cross_v)
        state.write_state("cross_mask", cross_mask)

        attn_mask = np.full((1, 1, 1, S_MAX), -1e4, dtype=np.float32)
        attn_mask[..., 0] = 0.0
        onehot = np.zeros((1, 1, S_MAX, 1), dtype=np.float32)
        onehot[0, 0, 0, 0] = 1.0

        BOS, EOS = 1, 2
        tokens = [BOS]
        per_token_times = []

        for step in range(min(S_MAX, 200)):
            tk0 = time.perf_counter()
            cos, sin = self.cos_tables[step], self.sin_tables[step]
            out = self.decoder.predict(
                {
                    "input_ids": np.array([[tokens[-1]]], dtype=np.int32),
                    "attn_mask": attn_mask,
                    "cos": cos,
                    "sin": sin,
                    "write_onehot": onehot,
                },
                state=state,
            )
            next_tok = int(np.asarray(out["logits"])[0, 0].argmax())
            tokens.append(next_tok)
            per_token_times.append(time.perf_counter() - tk0)

            if next_tok == EOS:
                break
            nxt = step + 1
            if nxt < S_MAX:
                attn_mask[..., nxt] = 0.0
                onehot = np.zeros((1, 1, S_MAX, 1), dtype=np.float32)
                onehot[0, 0, nxt, 0] = 1.0

        times["decoder_loop"] = time.perf_counter() - t0
        times["decoder_steps"] = len(per_token_times)
        times["decoder_per_token_mean"] = float(np.mean(per_token_times)) if per_token_times else 0
        times["decoder_per_token_max"] = float(np.max(per_token_times)) if per_token_times else 0
        times["decoder_per_token_min"] = float(np.min(per_token_times)) if per_token_times else 0

        # 5. Token decode.
        t0 = time.perf_counter()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        times["token_decode"] = time.perf_counter() - t0

        return text, times

    def transcribe_chunked(self, audio: np.ndarray, verbose: bool = False):
        """Transcribe arbitrary-length audio using 10s chunks with 2s overlap."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            if audio.shape[0] != 1:
                raise ValueError(f"Expected mono audio with shape (1, n), got {audio.shape}")
            audio = audio[0]
        elif audio.ndim != 1:
            raise ValueError(f"Expected 1D mono audio, got shape {audio.shape}")

        if len(audio) <= ENC_WINDOW:
            return self.transcribe(audio)

        step = ENC_WINDOW - OVERLAP_SAMPLES
        full_text = ""
        prev_text = ""
        previous_end = 0
        chunk_times = []
        chunk_index = 0

        for start in range(0, len(audio), step):
            end = min(start + ENC_WINDOW, len(audio))
            new_audio_samples = end - previous_end
            if start > 0 and new_audio_samples < MIN_NEW_AUDIO_SAMPLES:
                break
            if end - start < MIN_NEW_AUDIO_SAMPLES:
                break

            chunk = audio[start:end]
            text, times = self.transcribe(chunk)
            chunk_times.append(times)

            text = text.strip()
            new_part = dedup_overlap(prev_text, text) if prev_text else text
            if new_part:
                full_text = f"{full_text} {new_part}".strip()

            if verbose:
                t_s = start / SAMPLE_RATE
                t_e = end / SAMPLE_RATE
                clip = text[:50].replace("\n", " ")
                print(f"  chunk {chunk_index:2d} ({t_s:.0f}s-{t_e:.0f}s): \"{clip}...\"")

            prev_text = text
            previous_end = end
            chunk_index += 1

        if not chunk_times:
            return "", {
                "preprocess": 0.0,
                "encoder": 0.0,
                "kv_projection": 0.0,
                "decoder_loop": 0.0,
                "token_decode": 0.0,
                "decoder_steps": 0,
                "decoder_per_token_mean": 0.0,
                "decoder_per_token_max": 0.0,
                "decoder_per_token_min": 0.0,
                "chunks": 0,
            }

        times = {}
        for key in ["preprocess", "encoder", "kv_projection", "decoder_loop", "token_decode"]:
            times[key] = sum(t.get(key, 0.0) for t in chunk_times)

        total_steps = sum(int(t.get("decoder_steps", 0)) for t in chunk_times)
        times["decoder_steps"] = total_steps
        if total_steps:
            weighted_sum = sum(
                t.get("decoder_per_token_mean", 0.0) * int(t.get("decoder_steps", 0))
                for t in chunk_times
            )
            times["decoder_per_token_mean"] = weighted_sum / total_steps
            times["decoder_per_token_max"] = max(t.get("decoder_per_token_max", 0.0) for t in chunk_times)
            times["decoder_per_token_min"] = min(
                t.get("decoder_per_token_min", 0.0)
                for t in chunk_times
                if int(t.get("decoder_steps", 0)) > 0
            )
        else:
            times["decoder_per_token_mean"] = 0.0
            times["decoder_per_token_max"] = 0.0
            times["decoder_per_token_min"] = 0.0
        times["chunks"] = len(chunk_times)

        return full_text.strip(), times

    def warmup(self):
        """Pre-warm the ANE with a dummy inference."""
        print("  Warming up ANE…")
        audio = np.zeros((1, ENC_WINDOW), dtype=np.float32)
        _ = self.encoder.predict({self.enc_in: audio})


def main():
    import argparse

    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("must be >= 1")
        return parsed

    def positive_float(value: str) -> float:
        parsed = float(value)
        min_duration = 1 / SAMPLE_RATE
        if not math.isfinite(parsed):
            raise argparse.ArgumentTypeError("must be finite")
        if parsed < min_duration:
            raise argparse.ArgumentTypeError(f"must be >= {min_duration:.8f} seconds")
        return parsed

    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?")
    ap.add_argument("--iterations", type=positive_int, default=3)
    ap.add_argument("--duration", type=positive_float, default=3.0,
                    help="Synthetic audio duration in seconds")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    args = ap.parse_args()

    if args.audio:
        try:
            inspect_audio_wav(args.audio)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)

    bench = MoonshineBench()
    if not bench.load():
        sys.exit(1)

    bench.warmup()

    # Load or generate audio.
    if args.audio:
        audio = load_audio_wav(args.audio)
        label = os.path.basename(args.audio)
    else:
        audio = generate_test_audio(args.duration)
        label = f"synthetic_{args.duration}s"

    duration_s = len(audio) / SAMPLE_RATE
    if not args.json:
        print(f"\nAudio: {label} ({duration_s:.1f}s, {len(audio)} samples)")
        print(f"Iterations: {args.iterations}")
        print()

    all_times = []
    for i in range(args.iterations):
        text, times = bench.transcribe_chunked(audio.copy(), verbose=not args.json)
        all_times.append(times)

    # Aggregate.
    keys = ["preprocess", "encoder", "kv_projection", "decoder_loop",
            "token_decode", "decoder_steps", "decoder_per_token_mean",
            "decoder_per_token_max", "decoder_per_token_min", "chunks"]

    agg = {}
    for k in keys:
        vals = [t.get(k, 0) for t in all_times]
        if k in ("decoder_steps", "chunks"):
            agg[k] = int(np.mean(vals))
        else:
            agg[k] = {
                "mean_ms": round(float(np.mean(vals)) * 1000, 1),
                "min_ms": round(float(np.min(vals)) * 1000, 1),
                "max_ms": round(float(np.max(vals)) * 1000, 1),
                "std_ms": round(float(np.std(vals)) * 1000, 1),
            }

    total = sum(agg[k]["mean_ms"] if isinstance(agg[k], dict) else 0
                for k in ["preprocess", "encoder", "kv_projection", "decoder_loop", "token_decode"])
    agg["total_ms"] = round(total, 1)

    if args.json:
        agg["text"] = text
        agg["audio_s"] = round(duration_s, 1)
        print(json.dumps(agg, indent=2))
    else:
        print(f"{'Stage':<30} {'Mean (ms)':>10} {'Min':>8} {'Max':>8} {'Std':>8}")
        print("-" * 64)
        for k in ["preprocess", "encoder", "kv_projection", "decoder_loop", "token_decode"]:
            d = agg[k]
            print(f"{k:<30} {d['mean_ms']:>8.1f}  {d['min_ms']:>6.1f}  {d['max_ms']:>6.1f}  {d['std_ms']:>6.1f}")
        print("-" * 64)
        print(f"{'TOTAL':<30} {total:>8.1f} ms")
        print()
        print(f"Decoder steps: {agg['decoder_steps']}")
        print(f"Per-token mean: {agg['decoder_per_token_mean']['mean_ms']:.1f} ms")
        print(f"Per-token range: {agg['decoder_per_token_min']['mean_ms']:.1f} – {agg['decoder_per_token_max']['mean_ms']:.1f} ms")
        print(f"Text: '{text}'")


if __name__ == "__main__":
    main()
