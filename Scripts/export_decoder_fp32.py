#!/usr/bin/env python3
"""
Re-export the Moonshine decoder with two changes vs export_models.py:

  1. State dtype = fp32 (was fp16). Eliminates any cast loss in the
     read-blend-write cycle.

  2. Blend rewritten in additive form:
        sk_new = sk_old + (k_new - sk_old) * w
     instead of:
        sk_new = sk_old * (1 - w) + k_new * w
     mathematically identical, but the tracer captures different ops and
     this form preserves sk_old at the masked positions more cleanly.

  3. Cast `k_new * w` written as `k_new.expand(...) * w` so the broadcast
     is explicit, removing trace ambiguity.

  4. Write the buffer back via index_copy-like semantics (still in-place,
     but with a clearer pattern).

If this fixes the multi-step divergence, the issue was precision/trace fragility.
If it doesn't, the bug is elsewhere (RoPE indexing, attn_mask format, etc).
"""
import warnings
warnings.filterwarnings("ignore")

import sys, os, json, shutil
import numpy as np
import torch
import coremltools as ct
from pathlib import Path
from transformers import AutoConfig
from transformers.models.moonshine.modeling_moonshine import MoonshineForConditionalGeneration

MODEL_NAME = "UsefulSensors/moonshine-tiny"
OUTPUT_DIR = Path.home() / ".cache" / "moonshine-coreml" / "tiny-streaming"

NL, H, D, HID = 6, 8, 36, 288
S_MAX, S_ENC_MAX, ROT_DIM = 128, 500, 32


def rotate_half(x):
    x1, x2 = x[..., :ROT_DIM // 2], x[..., ROT_DIM // 2:]
    return torch.cat([-x2, x1], dim=-1)


class DecoderFP32(torch.nn.Module):
    """Same logic, fp32 states, additive blend."""
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config._attn_implementation = "eager"
        full = MoonshineForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)
        self.decoder = full.model.decoder
        self.proj_out = full.proj_out
        for p in self.parameters():
            p.requires_grad = False

        # fp32 state buffers.
        self.register_buffer("cross_k", torch.zeros(NL, 1, H, S_ENC_MAX, D, dtype=torch.float32))
        self.register_buffer("cross_v", torch.zeros(NL, 1, H, S_ENC_MAX, D, dtype=torch.float32))
        self.register_buffer("cross_mask", torch.zeros(1, 1, 1, S_ENC_MAX, dtype=torch.float32))
        self.register_buffer("self_k", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float32))
        self.register_buffer("self_v", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float32))
        self.eval()

    def forward(self, input_ids, attn_mask, cos, sin, write_onehot):
        B, T = 1, 1
        hs = self.decoder.embed_tokens(input_ids)

        for i, layer in enumerate(self.decoder.layers):
            # ── Self-attention ──────────────────────────────────────────────
            res = hs
            h = layer.input_layernorm(hs)
            q = layer.self_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            k = layer.self_attn.k_proj(h).view(B, T, H, D).transpose(1, 2)
            v = layer.self_attn.v_proj(h).view(B, T, H, D).transpose(1, 2)

            # RoPE on first ROT_DIM dims.
            qr = q[..., :ROT_DIM] * cos + rotate_half(q[..., :ROT_DIM]) * sin
            kr = k[..., :ROT_DIM] * cos + rotate_half(k[..., :ROT_DIM]) * sin
            q = torch.cat([qr, q[..., ROT_DIM:]], dim=-1)
            k = torch.cat([kr, k[..., ROT_DIM:]], dim=-1)

            # Read prior state (fp32, no cast).
            sk_old = self.self_k[i]              # [1, H, S_MAX, D]
            sv_old = self.self_v[i]
            w = write_onehot                     # [1, 1, S_MAX, 1] in 0/1

            # Additive blend: equivalent to sk_old*(1-w) + k_broadcast*w but
            # more numerically stable when w is exactly 0 or 1.
            # k is [1, H, 1, D]; broadcasting against [1, 1, S_MAX, 1] gives
            # [1, H, S_MAX, D] with k replicated along the S_MAX axis.
            k_broadcast = k.expand(1, H, S_MAX, D)
            v_broadcast = v.expand(1, H, S_MAX, D)
            sk_new = sk_old + (k_broadcast - sk_old) * w
            sv_new = sv_old + (v_broadcast - sv_old) * w

            # Attention against the freshly updated state.
            attn = q @ sk_new.transpose(-2, -1) * (D ** -0.5) + attn_mask
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ sv_new).transpose(1, 2).reshape(B, T, -1)
            hs = res + layer.self_attn.o_proj(out)

            # Persist the update (in-place; tracer-friendly).
            self.self_k.data[i] = sk_new
            self.self_v.data[i] = sv_new

            # ── Cross-attention ─────────────────────────────────────────────
            res = hs
            h = layer.post_attention_layernorm(hs)
            q = layer.encoder_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            ck = self.cross_k[i]
            cv = self.cross_v[i]
            cm = self.cross_mask
            attn = q @ ck.transpose(-2, -1) * (D ** -0.5) + cm
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ cv).transpose(1, 2).reshape(B, T, -1)
            hs = res + layer.encoder_attn.o_proj(out)

            # ── MLP ─────────────────────────────────────────────────────────
            res = hs
            h = layer.final_layernorm(hs)
            mlp = layer.mlp.fc1(h)
            up, gate = mlp.chunk(2, dim=-1)
            hs = res + layer.mlp.fc2(torch.nn.functional.silu(gate) * up)

        hs = self.decoder.norm(hs)
        return self.proj_out(hs)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building decoder model (fp32 states, additive blend)…")
    model = DecoderFP32().eval()
    with torch.no_grad():
        model.cross_mask.data = torch.full((1, 1, 1, S_ENC_MAX), -100.0, dtype=torch.float32)
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
    print("Tracing…")
    with torch.no_grad():
        traced = torch.jit.trace(model, args)

    print("Converting to CoreML (this takes a minute)…")
    stateful = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids",    shape=(1, 1),                  dtype=np.int32),
            ct.TensorType(name="attn_mask",    shape=(1, 1, 1, S_MAX),        dtype=np.float32),
            ct.TensorType(name="cos",          shape=(1, 1, 1, ROT_DIM),      dtype=np.float32),
            ct.TensorType(name="sin",          shape=(1, 1, 1, ROT_DIM),      dtype=np.float32),
            ct.TensorType(name="write_onehot", shape=(1, 1, S_MAX, 1),        dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        states=[
            ct.StateType(name="cross_k",    wrapped_type=ct.TensorType(
                shape=(NL, 1, H, S_ENC_MAX, D), dtype=np.float32)),
            ct.StateType(name="cross_v",    wrapped_type=ct.TensorType(
                shape=(NL, 1, H, S_ENC_MAX, D), dtype=np.float32)),
            ct.StateType(name="cross_mask", wrapped_type=ct.TensorType(
                shape=(1, 1, 1, S_ENC_MAX), dtype=np.float32)),
            ct.StateType(name="self_k",     wrapped_type=ct.TensorType(
                shape=(NL, 1, H, S_MAX, D), dtype=np.float32)),
            ct.StateType(name="self_v",     wrapped_type=ct.TensorType(
                shape=(NL, 1, H, S_MAX, D), dtype=np.float32)),
        ],
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18,
    )

    path = OUTPUT_DIR / "decoder_fp32.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    stateful.save(str(path))
    print(f"✓ Saved → {path}")


if __name__ == "__main__":
    main()
