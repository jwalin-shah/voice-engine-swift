#!/usr/bin/env python3
"""
Export Moonshine-streaming-medium (v2) encoder + stateful decoder to CoreML .mlpackage.

Requirements:
    pip install --break-system-packages "transformers @ git+https://github.com/huggingface/transformers.git@main"
    pip install --break-system-packages coremltools torch numpy sentencepiece

Output:
    ~/.cache/moonshine-coreml/streaming-medium/
        encoder.mlpackage
        decoder_stateful.mlpackage
        weights/
            config.json
            cos_tables.f32 / sin_tables.f32
            layer{N}_k_weight.f32 / layer{N}_v_weight.f32
            pos_emb.f32
            decoder_proj.f32
        id_to_piece.json
"""

import os, sys, json, shutil
import numpy as np
import torch
import coremltools as ct
from pathlib import Path
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq

MODEL_NAME = "UsefulSensors/moonshine-streaming-medium"
OUTPUT_DIR = Path.home() / ".cache" / "moonshine-coreml" / "streaming-medium"

# Architecture constants (verified from config.json)
NL = 14          # decoder layers
H = 10           # attention heads
D = 64           # head dimension
DEC_HID = 640    # decoder hidden size
ENC_HID = 768    # encoder hidden size
S_MAX = 128      # max decoder steps
S_ENC_MAX = 500  # max encoder frames (10s at 50Hz)
ROT_DIM = 32     # rotary dimension (D * partial_rotary_factor = 64 * 0.5)
ENC_WINDOW = 160000  # 10s at 16kHz


# ── Model loading helper ─────────────────────────────────────────────────────

def load_model():
    """Load the v2 model with eager attention (required for JIT trace)."""
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config._attn_implementation = "eager"
    config.encoder_config._attn_implementation = "eager"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME, config=config, trust_remote_code=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ── Encoder Export ───────────────────────────────────────────────────────────

def export_encoder():
    """Export the full MoonshineStreamingEncoder (includes preprocessor).

    Note: monkey-patches torch.asinh -> log(x+sqrt(x^2+1)) because coremltools
    doesn't support asinh natively. The formula is numerically equivalent.
    """
    print("-> Exporting encoder (v2 streaming-medium)...")

    model = load_model()
    encoder = model.model.encoder
    encoder.eval()

    # Monkey-patch asinh for coremltools compatibility.
    _orig_asinh = torch.asinh
    torch.asinh = lambda x: torch.log(x + torch.sqrt(x * x + 1))

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.encoder = enc
        def forward(self, x):
            return self.encoder(x).last_hidden_state

    wrapped = EncoderWrapper(encoder).eval()

    # Trace with 10s dummy input (max bucket).
    dummy = torch.randn(1, ENC_WINDOW)
    try:
        with torch.no_grad():
            traced = torch.jit.trace(wrapped, dummy)
            out = traced(dummy)
        print(f"   Traced encoder output: {out.shape} (expected [1, {S_ENC_MAX}, {ENC_HID}])")

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="audio", shape=(1, ENC_WINDOW), dtype=np.float32)],
            outputs=[ct.TensorType(name="hidden_states", dtype=np.float32)],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.iOS18,
        )

        path = OUTPUT_DIR / "encoder.mlpackage"
        if path.exists():
            shutil.rmtree(path)
        mlmodel.save(str(path))
        print(f"   Saved -> {path}")
    finally:
        torch.asinh = _orig_asinh


# ── Rotate half (GPT-J interleaved) ─────────────────────────────────────────

def rotate_half(x):
    """GPT-J style interleaved rotary (even/odd pairs -> rotate)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


# ── Decoder Export ───────────────────────────────────────────────────────────

class DecoderWithCrossMaskV2(torch.nn.Module):
    """Stateful decoder wrapper for v2 streaming-medium.

    Stores pre-computed cross-KV (already projected from 768->640 via
    decoder.proj + pos_emb) in state. Self-attention KV cache accumulates
    step by step.
    """

    def __init__(self):
        super().__init__()
        model = load_model()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
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

        sk_state = self.self_k.to(torch.float32)
        sv_state = self.self_v.to(torch.float32)
        ck_state = self.cross_k.to(torch.float32)
        cv_state = self.cross_v.to(torch.float32)
        cm_state = self.cross_mask.to(torch.float32)
        w = write_onehot.to(torch.float32)

        new_sk_layers = []
        new_sv_layers = []

        for i, layer in enumerate(self.decoder.layers):
            # ── Self-attention ─────────────────────────────────────────────
            res = hs
            h = layer.input_layernorm(hs)
            q = layer.self_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            k = layer.self_attn.k_proj(h).view(B, T, H, D).transpose(1, 2)
            v = layer.self_attn.v_proj(h).view(B, T, H, D).transpose(1, 2)

            # RoPE (interleaved variant - GPT-J style).
            qr = q[..., :ROT_DIM] * cos + rotate_half(q[..., :ROT_DIM]) * sin
            kr = k[..., :ROT_DIM] * cos + rotate_half(k[..., :ROT_DIM]) * sin
            q = torch.cat([qr, q[..., ROT_DIM:]], dim=-1)
            k = torch.cat([kr, k[..., ROT_DIM:]], dim=-1)

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

            # ── MLP (SwiGLU: fc1 outputs 2*intermediate, chunk into gate/up) ─
            res = hs
            h = layer.final_layernorm(hs)
            mlp = layer.mlp.fc1(h)
            up, gate = mlp.chunk(2, dim=-1)
            hs = res + layer.mlp.fc2(torch.nn.functional.silu(gate) * up)

        full_new_sk = torch.stack(new_sk_layers, dim=0).to(torch.float16)
        full_new_sv = torch.stack(new_sv_layers, dim=0).to(torch.float16)
        self.self_k[:] = full_new_sk
        self.self_v[:] = full_new_sv

        hs = self.decoder.norm(hs)
        return self.proj_out(hs)


def export_decoder():
    print("-> Exporting stateful decoder with cross_mask (v2 medium)...")

    model_wrapper = DecoderWithCrossMaskV2().eval()

    with torch.no_grad():
        model_wrapper.cross_mask.data = torch.full(
            (1, 1, 1, S_ENC_MAX), -100.0, dtype=torch.float16
        )
        model_wrapper.cross_mask.data[..., :250] = 0.0

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
        traced = torch.jit.trace(model_wrapper, args)
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
        compute_units=ct.ComputeUnit.CPU_ONLY,  # Stateful tensors -> CPU
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
    )

    path = OUTPUT_DIR / "decoder_stateful.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    stateful.save(str(path))
    print(f"   Saved -> {path}")


# ── KV Weights Extraction ───────────────────────────────────────────────────

def extract_kv_weights():
    """Extract cross-attention K/V projection weights from the v2 model.

    For v2, we merge per-layer cross-attn k_proj/v_proj with decoder.proj,
    and compute per-position biases from pos_emb.
    """
    print("-> Extracting cross-attention KV weights (v2 medium)...")

    model = load_model()
    decoder = model.model.decoder
    proj_weight = decoder.proj.weight.detach().numpy().astype(np.float32)    # [DEC_HID, ENC_HID]
    pos_emb_weight = decoder.pos_emb.weight.detach().numpy().astype(np.float32)  # [4096, ENC_HID]

    # Build RoPE tables for positions 0..S_MAX-1 from the decoder's rotary_emb.
    dec_rope = decoder.rotary_emb
    pos_ids = torch.arange(S_MAX).unsqueeze(0)
    with torch.no_grad():
        cos_raw, sin_raw = dec_rope(torch.zeros(1, S_MAX, DEC_HID), position_ids=pos_ids)
    half = cos_raw.shape[-1] // 2
    cos_used = cos_raw[..., :half].repeat_interleave(2, dim=-1)
    sin_used = sin_raw[..., :half].repeat_interleave(2, dim=-1)
    cos_tables = [cos_used[0, pos].detach().numpy().astype(np.float32) for pos in range(S_MAX)]
    sin_tables = [sin_used[0, pos].detach().numpy().astype(np.float32) for pos in range(S_MAX)]

    kw_list, vw_list = [], []
    pos_bias_k_list, pos_bias_v_list = [], []

    for layer in decoder.layers:
        kp_w = layer.encoder_attn.k_proj.weight.detach().numpy().astype(np.float32)  # [DEC_HID, DEC_HID]
        vp_w = layer.encoder_attn.v_proj.weight.detach().numpy().astype(np.float32)  # [DEC_HID, DEC_HID]

        # merged = k_proj @ proj  (result is [DEC_HID, ENC_HID])
        merged_k = kp_w @ proj_weight   # [DEC_HID, DEC_HID] @ [DEC_HID, ENC_HID] = [DEC_HID, ENC_HID]
        merged_v = vp_w @ proj_weight
        kw_list.append(merged_k)
        vw_list.append(merged_v)

        # pos_bias[i][pos] = k_proj_i @ proj @ pos_emb[pos]
        pe = pos_emb_weight[:S_ENC_MAX, :]  # [S_ENC_MAX, ENC_HID]
        pb_k = (merged_k @ pe.T).astype(np.float32)  # [DEC_HID, S_ENC_MAX]
        pb_v = (merged_v @ pe.T).astype(np.float32)
        pos_bias_k_list.append(pb_k.T)   # [S_ENC_MAX, DEC_HID]
        pos_bias_v_list.append(pb_v.T)

    out = {
        "NL": np.int32(NL), "H": np.int32(H), "D": np.int32(D),
        "DEC_HID": np.int32(DEC_HID), "ENC_HID": np.int32(ENC_HID),
        "S_MAX": np.int32(S_MAX), "S_ENC_MAX": np.int32(S_ENC_MAX),
        "ROT_DIM": np.int32(ROT_DIM),
        "cos_tables": np.stack(cos_tables),  # [S_MAX, ROT_DIM]
        "sin_tables": np.stack(sin_tables),  # [S_MAX, ROT_DIM]
        "pos_emb": pos_emb_weight[:S_ENC_MAX, :],  # [S_ENC_MAX, ENC_HID]
        "decoder_proj": proj_weight,                 # [DEC_HID, ENC_HID]
    }
    for i in range(NL):
        out[f"layer{i}_k_weight"] = kw_list[i]       # [DEC_HID, ENC_HID]
        out[f"layer{i}_v_weight"] = vw_list[i]       # [DEC_HID, ENC_HID]
        out[f"layer{i}_pos_bias_k"] = pos_bias_k_list[i]  # [S_ENC_MAX, DEC_HID]
        out[f"layer{i}_pos_bias_v"] = pos_bias_v_list[i]  # [S_ENC_MAX, DEC_HID]

    npz_path = OUTPUT_DIR / "cross_kv_weights.npz"
    np.savez_compressed(str(npz_path), **out)
    print(f"   Saved -> {npz_path}")

    _write_flat_binaries(out)


def _write_flat_binaries(data: dict):
    """Write per-layer weights as flat float32 little-endian binaries."""
    bin_dir = OUTPUT_DIR / "weights"
    bin_dir.mkdir(exist_ok=True)

    config = {
        "NL": int(data["NL"]), "H": int(data["H"]), "D": int(data["D"]),
        "DEC_HID": int(data["DEC_HID"]), "ENC_HID": int(data["ENC_HID"]),
        "S_MAX": int(data["S_MAX"]), "S_ENC_MAX": int(data["S_ENC_MAX"]),
        "ROT_DIM": int(data["ROT_DIM"]),
    }
    with open(bin_dir / "config.json", "w") as f:
        json.dump(config, f)

    data["cos_tables"].astype(np.float32).tofile(str(bin_dir / "cos_tables.f32"))
    data["sin_tables"].astype(np.float32).tofile(str(bin_dir / "sin_tables.f32"))

    data["pos_emb"].astype(np.float32).tofile(str(bin_dir / "pos_emb.f32"))
    data["decoder_proj"].astype(np.float32).tofile(str(bin_dir / "decoder_proj.f32"))

    for i in range(NL):
        data[f"layer{i}_k_weight"].astype(np.float32).tofile(
            str(bin_dir / f"layer{i}_k_weight.f32"))
        data[f"layer{i}_v_weight"].astype(np.float32).tofile(
            str(bin_dir / f"layer{i}_v_weight.f32"))
        data[f"layer{i}_pos_bias_k"].astype(np.float32).tofile(
            str(bin_dir / f"layer{i}_pos_bias_k.f32"))
        data[f"layer{i}_pos_bias_v"].astype(np.float32).tofile(
            str(bin_dir / f"layer{i}_pos_bias_v.f32"))

    print(f"   Flat binaries -> {bin_dir}")


# ── Tokenizer ──────────────────────────────────────────────────────────────

def copy_tokenizer():
    """Copy tokenizer artifacts. V2 uses the same tokenizer as v1 (same vocab)."""
    print("-> Copying tokenizer (from v2 medium model)...")
    from transformers import AutoTokenizer
    import tempfile

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    dst_spm = OUTPUT_DIR / "sentencepiece.bpe.model"
    spm_path = getattr(tok, "vocab_file", None)
    if spm_path and os.path.exists(spm_path):
        shutil.copy(spm_path, dst_spm)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tok.save_pretrained(tmpdir)
            for fname in os.listdir(tmpdir):
                if fname.endswith(".model"):
                    shutil.copy(os.path.join(tmpdir, fname), dst_spm)
                    break

    _convert_tokenizer_json(tok, OUTPUT_DIR / "id_to_piece.json")
    print(f"   Tokenizer saved")


def _convert_tokenizer_json(tok, out_path):
    """Convert tokenizer to id_to_piece.json for Swift."""
    vocab = tok.get_vocab()
    id_to_piece = {}
    for piece, idx in vocab.items():
        id_to_piece[str(idx)] = piece
    with open(out_path, "w") as f:
        json.dump({"id_to_piece": id_to_piece}, f)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if "--encoder-only" not in sys.argv and "--decoder-only" not in sys.argv \
       and "--weights-only" not in sys.argv and "--tokenizer-only" not in sys.argv:
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

    print(f"\nAll artifacts in {OUTPUT_DIR}")
    print("  encoder.mlpackage")
    print("  decoder_stateful.mlpackage")
    print("  cross_kv_weights.npz")
    print("  weights/  (flat f32 binaries)")
    print("  sentencepiece.bpe.model")
    print("  id_to_piece.json")


if __name__ == "__main__":
    main()
