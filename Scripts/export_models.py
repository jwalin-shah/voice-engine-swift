#!/usr/bin/env python3
"""
Export Moonshine-tiny encoder + stateful decoder to CoreML .mlpackage.

Requirements:
    pip install coremltools transformers moonshine-onnx torch numpy sentencepiece

Output:
    ~/.cache/moonshine-coreml/tiny-streaming/
        encoder.mlpackage
        decoder_stateful.mlpackage
        cross_kv_weights.npz
        sentencepiece.bpe.model
        config.json
"""

import os, sys, json, shutil
import numpy as np
import torch
import coremltools as ct
from pathlib import Path
from transformers import AutoConfig
from transformers.models.moonshine.modeling_moonshine import MoonshineForConditionalGeneration
import sentencepiece as spm

MODEL_NAME = "UsefulSensors/moonshine-tiny"
OUTPUT_DIR = Path.home() / ".cache" / "moonshine-coreml" / "tiny-streaming"

# Architecture constants
NL, H, D, HID = 6, 8, 36, 288
S_MAX, S_ENC_MAX, ROT_DIM = 128, 500, 32
ENC_WINDOW = 160000         # 10 s at 16 kHz
BUCKET_SIZES = [16000, 48000, 80000, 160000]  # 1, 3, 5, 10 s


# ── Encoder Export ───────────────────────────────────────────────────────────

def export_encoder():
    """Export the full Moonshine encoder (audio frontend + transformer encoder)."""
    print("→ Exporting encoder…")

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config._attn_implementation = "eager"
    full = MoonshineForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)
    encoder = full.model.encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Wrap encoder to extract last_hidden_state from the dict output.
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.encoder = enc
        def forward(self, x):
            return self.encoder(x).last_hidden_state

    wrapped = EncoderWrapper(encoder).eval()

    # Trace with a 10 s dummy input (max bucket).
    dummy = torch.randn(1, ENC_WINDOW)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="audio", shape=(1, ENC_WINDOW), dtype=np.float32)],
        outputs=[ct.TensorType(name="hidden_states", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )

    path = OUTPUT_DIR / "encoder.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    mlmodel.save(str(path))
    print(f"   Saved → {path}")


# ── Decoder Export ────────────────────────────────────────────────────────────

def rotate_half(x):
    # HF Moonshine uses GPT-J-style INTERLEAVED rotary (even/odd pairs),
    # NOT split halves. Must match HF apply_rotary_pos_emb in
    # transformers/models/moonshine/modeling_moonshine.py.
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class DecoderWithCrossMask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config._attn_implementation = "eager"
        full = MoonshineForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)
        self.decoder = full.model.decoder
        self.proj_out = full.proj_out
        for p in self.parameters():
            p.requires_grad = False
        # fp16 state buffers (ANE constraint).
        self.register_buffer("cross_k", torch.zeros(NL, 1, H, S_ENC_MAX, D, dtype=torch.float16))
        self.register_buffer("cross_v", torch.zeros(NL, 1, H, S_ENC_MAX, D, dtype=torch.float16))
        self.register_buffer("cross_mask", torch.zeros(1, 1, 1, S_ENC_MAX, dtype=torch.float16))
        self.register_buffer("self_k", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float16))
        self.register_buffer("self_v", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float16))
        self.eval()

    def forward(self, input_ids, attn_mask, cos, sin, write_onehot):
        B, T = 1, 1
        hs = self.decoder.embed_tokens(input_ids)

        # Read state ONCE at start (cast to fp32 for compute precision).
        sk_state = self.self_k.to(torch.float32)
        sv_state = self.self_v.to(torch.float32)
        ck_state = self.cross_k.to(torch.float32)
        cv_state = self.cross_v.to(torch.float32)
        cm_state = self.cross_mask.to(torch.float32)
        w = write_onehot.to(torch.float32)

        # Collect new self_k/self_v per layer; we apply the whole-buffer
        # copy_() at the end so the JIT trace captures a single state write
        # per buffer (CoreML maps this to coreml_update_state).
        new_sk_layers = []
        new_sv_layers = []

        for i, layer in enumerate(self.decoder.layers):
            # ── Self-attention ─────────────────────────────────────────────
            res = hs
            h = layer.input_layernorm(hs)
            q = layer.self_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            k = layer.self_attn.k_proj(h).view(B, T, H, D).transpose(1, 2)
            v = layer.self_attn.v_proj(h).view(B, T, H, D).transpose(1, 2)

            # RoPE (interleaved variant — see rotate_half).
            qr = q[..., :ROT_DIM] * cos + rotate_half(q[..., :ROT_DIM]) * sin
            kr = k[..., :ROT_DIM] * cos + rotate_half(k[..., :ROT_DIM]) * sin
            q = torch.cat([qr, q[..., ROT_DIM:]], dim=-1)
            k = torch.cat([kr, k[..., ROT_DIM:]], dim=-1)

            # Blend new k/v into the cache at the position selected by write_onehot.
            sk_new = sk_state[i] * (1 - w) + k * w
            sv_new = sv_state[i] * (1 - w) + v * w

            attn = q @ sk_new.transpose(-2, -1) * (D ** -0.5) + attn_mask
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ sv_new).transpose(1, 2).reshape(B, T, -1)
            hs = res + layer.self_attn.o_proj(out)

            new_sk_layers.append(sk_new)
            new_sv_layers.append(sv_new)

            # ── Cross-attention ────────────────────────────────────────────
            res = hs
            h = layer.post_attention_layernorm(hs)
            q = layer.encoder_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            attn = q @ ck_state[i].transpose(-2, -1) * (D ** -0.5) + cm_state
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ cv_state[i]).transpose(1, 2).reshape(B, T, -1)
            hs = res + layer.encoder_attn.o_proj(out)

            # ── MLP ────────────────────────────────────────────────────────
            res = hs
            h = layer.final_layernorm(hs)
            mlp = layer.mlp.fc1(h)
            up, gate = mlp.chunk(2, dim=-1)
            hs = res + layer.mlp.fc2(torch.nn.functional.silu(gate) * up)

        # Stack into [NL, 1, H, S_MAX, D] then write the FULL buffer in one
        # in-place op. torch.jit.trace + coremltools 8+ capture this as a
        # coreml_update_state op (verified: MIL graph contains 2 update_state
        # ops for self_k and self_v after this change).
        full_new_sk = torch.stack(new_sk_layers, dim=0).to(torch.float16)
        full_new_sv = torch.stack(new_sv_layers, dim=0).to(torch.float16)
        # Slice-assign the full buffer. coremltools' tensor-assignment pass
        # recognizes "buffer[:] = tensor" as a state write and emits
        # coreml_update_state.
        self.self_k[:] = full_new_sk
        self.self_v[:] = full_new_sv

        hs = self.decoder.norm(hs)
        return self.proj_out(hs)


def export_decoder():
    print("→ Exporting stateful decoder with cross_mask…")

    model = DecoderWithCrossMask().eval()

    # Pre-seed cross_mask so coremltools doesn't optimize away the path.
    with torch.no_grad():
        model.cross_mask.data = torch.full((1, 1, 1, S_ENC_MAX), -100.0, dtype=torch.float16)
        model.cross_mask.data[..., :250] = 0.0

    args = (
        torch.tensor([[1]], dtype=torch.int32),
        torch.full((1, 1, 1, S_MAX), -1e4, dtype=torch.float32).scatter_(
            -1, torch.zeros((1, 1, 1, 1), dtype=torch.long), 0.0),
        torch.zeros(1, 1, 1, ROT_DIM, dtype=torch.float32),
        torch.zeros(1, 1, 1, ROT_DIM, dtype=torch.float32),
        torch.zeros(1, 1, S_MAX, 1, dtype=torch.float32).scatter_(
            -1, torch.tensor([[[[0]]]]), 1.0),
    )
    with torch.no_grad():
        traced = torch.jit.trace(model, args)
    print("   JIT trace OK")

    stateful = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="attn_mask", shape=(1, 1, 1, S_MAX), dtype=np.float32),
            ct.TensorType(name="cos", shape=(1, 1, 1, ROT_DIM), dtype=np.float32),
            ct.TensorType(name="sin", shape=(1, 1, 1, ROT_DIM), dtype=np.float32),
            ct.TensorType(name="write_onehot", shape=(1, 1, S_MAX, 1), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        states=[
            ct.StateType(name="cross_k", wrapped_type=ct.TensorType(
                shape=(NL, 1, H, S_ENC_MAX, D), dtype=np.float16)),
            ct.StateType(name="cross_v", wrapped_type=ct.TensorType(
                shape=(NL, 1, H, S_ENC_MAX, D), dtype=np.float16)),
            ct.StateType(name="cross_mask", wrapped_type=ct.TensorType(
                shape=(1, 1, 1, S_ENC_MAX), dtype=np.float16)),
            ct.StateType(name="self_k", wrapped_type=ct.TensorType(
                shape=(NL, 1, H, S_MAX, D), dtype=np.float16)),
            ct.StateType(name="self_v", wrapped_type=ct.TensorType(
                shape=(NL, 1, H, S_MAX, D), dtype=np.float16)),
        ],
        compute_units=ct.ComputeUnit.CPU_ONLY,  # Stateful tensors → CPU
        minimum_deployment_target=ct.target.iOS18,
    )

    path = OUTPUT_DIR / "decoder_stateful.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    stateful.save(str(path))
    print(f"   Saved → {path}")


# ── KV Weights Extraction ───────────────────────────────────────────────────

def extract_kv_weights():
    """Extract cross-attention K/V projection weights from the HF model."""
    print("→ Extracting cross-attention KV weights…")

    config = AutoConfig.from_pretrained(MODEL_NAME)
    config._attn_implementation = "eager"
    full = MoonshineForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)

    # Build RoPE tables for positions 0..S_MAX-1 from the DECODER's rotary_emb,
    # not the encoder. Apply the same interleave transform HF's
    # apply_rotary_pos_emb does internally so the exported tables match what
    # rotate_half (interleaved version) expects.
    dec_rope = full.model.decoder.rotary_emb
    pos_ids = torch.arange(S_MAX).unsqueeze(0)
    with torch.no_grad():
        try:
            cos_raw, sin_raw = dec_rope(torch.zeros(1, S_MAX, HID), position_ids=pos_ids)
        except TypeError:
            cos_raw, sin_raw = dec_rope(torch.zeros(1, S_MAX, HID), pos_ids)
    # HF takes first half and repeat_interleaves to produce per-axis sinusoids.
    half = cos_raw.shape[-1] // 2
    cos_used = cos_raw[..., :half].repeat_interleave(2, dim=-1)  # [1, S_MAX, ROT_DIM]
    sin_used = sin_raw[..., :half].repeat_interleave(2, dim=-1)
    cos_tables = [cos_used[0, pos].detach().numpy().astype(np.float32) for pos in range(S_MAX)]
    sin_tables = [sin_used[0, pos].detach().numpy().astype(np.float32) for pos in range(S_MAX)]

    # Cross-attention K/V weights per decoder layer.
    kw_list, vw_list, kb_list, vb_list = [], [], [], []
    for layer in full.model.decoder.layers:
        kw_list.append(layer.encoder_attn.k_proj.weight.detach().numpy().astype(np.float32))
        vw_list.append(layer.encoder_attn.v_proj.weight.detach().numpy().astype(np.float32))
        kb = layer.encoder_attn.k_proj.bias
        vb = layer.encoder_attn.v_proj.bias
        kb_list.append(kb.detach().numpy().astype(np.float32) if kb is not None else None)
        vb_list.append(vb.detach().numpy().astype(np.float32) if vb is not None else None)

    out = {
        "NL": np.int32(NL), "H": np.int32(H), "D": np.int32(D),
        "HID": np.int32(HID), "S_MAX": np.int32(S_MAX),
        "S_ENC_MAX": np.int32(S_ENC_MAX), "ROT_DIM": np.int32(ROT_DIM),
        "cos_tables": np.stack(cos_tables),  # [S_MAX, ROT_DIM]
        "sin_tables": np.stack(sin_tables),  # [S_MAX, ROT_DIM]
    }
    for i in range(NL):
        out[f"layer{i}_k_weight"] = kw_list[i]
        out[f"layer{i}_v_weight"] = vw_list[i]
        if kb_list[i] is not None:
            out[f"layer{i}_k_bias"] = kb_list[i]
        if vb_list[i] is not None:
            out[f"layer{i}_v_bias"] = vb_list[i]

    npz_path = OUTPUT_DIR / "cross_kv_weights.npz"
    np.savez_compressed(str(npz_path), **out)
    print(f"   Saved → {npz_path}")

    # Also write flat binaries for direct Swift consumption.
    _write_flat_binaries(out)


def _write_flat_binaries(data: dict):
    """Write per-layer weights as flat float32 little-endian binaries."""
    bin_dir = OUTPUT_DIR / "weights"
    bin_dir.mkdir(exist_ok=True)

    # Config
    config = {
        "NL": int(data["NL"]), "H": int(data["H"]), "D": int(data["D"]),
        "HID": int(data["HID"]), "S_MAX": int(data["S_MAX"]),
        "S_ENC_MAX": int(data["S_ENC_MAX"]), "ROT_DIM": int(data["ROT_DIM"]),
    }
    with open(bin_dir / "config.json", "w") as f:
        json.dump(config, f)

    # RoPE tables
    data["cos_tables"].astype(np.float32).tofile(str(bin_dir / "cos_tables.f32"))
    data["sin_tables"].astype(np.float32).tofile(str(bin_dir / "sin_tables.f32"))

    # Per-layer K/V weights
    for i in range(int(data["NL"])):
        data[f"layer{i}_k_weight"].astype(np.float32).tofile(
            str(bin_dir / f"layer{i}_k_weight.f32"))
        data[f"layer{i}_v_weight"].astype(np.float32).tofile(
            str(bin_dir / f"layer{i}_v_weight.f32"))
        if f"layer{i}_k_bias" in data:
            data[f"layer{i}_k_bias"].astype(np.float32).tofile(
                str(bin_dir / f"layer{i}_k_bias.f32"))
        if f"layer{i}_v_bias" in data:
            data[f"layer{i}_v_bias"].astype(np.float32).tofile(
                str(bin_dir / f"layer{i}_v_bias.f32"))

    print(f"   Flat binaries → {bin_dir}")


# ── Tokenizer ──────────────────────────────────────────────────────────────

def copy_tokenizer():
    """Copy the sentencepiece model from the HF cache."""
    print("→ Copying tokenizer…")
    from transformers import AutoTokenizer
    import tempfile
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    dst = OUTPUT_DIR / "sentencepiece.bpe.model"
    # PreTrainedTokenizerFast may lack vocab_file in newer transformers;
    # fall back to save_pretrained + discover the .model file.
    spm_path = getattr(tok, "vocab_file", None)
    if spm_path and os.path.exists(spm_path):
        shutil.copy(spm_path, dst)
        print(f"   Saved → {dst}")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        tok.save_pretrained(tmpdir)
        for fname in os.listdir(tmpdir):
            if fname.endswith(".model"):
                spm_path = os.path.join(tmpdir, fname)
                shutil.copy(spm_path, dst)
                print(f"   Saved → {dst}")
                return
    print("   WARNING: sentencepiece model not found in tokenizer")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if "--encoder-only" not in sys.argv and "--decoder-only" not in sys.argv:
        export_encoder()
        export_decoder()
        extract_kv_weights()
        copy_tokenizer()
    else:
        if "--encoder-only" in sys.argv:
            export_encoder()
        if "--decoder-only" in sys.argv:
            export_decoder()
        if "--weights-only" in sys.argv:
            extract_kv_weights()
        if "--tokenizer-only" in sys.argv:
            copy_tokenizer()

    print(f"\n✓ All artifacts in {OUTPUT_DIR}")
    print("  encoder.mlpackage")
    print("  decoder_stateful.mlpackage")
    print("  cross_kv_weights.npz")
    print("  weights/  (flat f32 binaries)")
    print("  sentencepiece.bpe.model")


if __name__ == "__main__":
    main()