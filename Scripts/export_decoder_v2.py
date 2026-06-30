#!/usr/bin/env python3
"""
Export Moonshine-tiny stateful decoder where cross-KV is passed as INPUTS
(not states). Only self-attention KV is stateful.

This avoids the CoreML state-write API problem in Swift. Cross-KV is
computed once per utterance (encoder + matmul) and passed in on every
decoder step — negligible overhead vs the model computation.

All-float32 to avoid coremltools validator issues with mixed precision.
"""

import os, sys, shutil
import numpy as np, torch, coremltools as ct
from pathlib import Path
from transformers import AutoConfig
from transformers.models.moonshine.modeling_moonshine import MoonshineForConditionalGeneration

MODEL_NAME = "UsefulSensors/moonshine-tiny"
OUTPUT_DIR = Path.home() / ".cache" / "moonshine-coreml" / "tiny-streaming"

NL, H, D, HID = 6, 8, 36, 288
S_MAX, S_ENC_MAX, ROT_DIM = 128, 500, 32


def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class DecoderWithCrossInputs(torch.nn.Module):
    """
    Cross-k/v/mask are INPUTS, not states.
    Self-attention k/v are states (accumulate per step).
    All-float32 for clean CoreML export.
    """
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config._attn_implementation = "eager"
        full = MoonshineForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)
        self.decoder = full.model.decoder
        self.proj_out = full.proj_out
        for p in self.parameters():
            p.requires_grad = False
        # All-float32 states
        self.register_buffer("self_k", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float32))
        self.register_buffer("self_v", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float32))
        self.eval()

    def forward(self, input_ids, attn_mask, cos, sin, write_onehot,
                cross_k, cross_v, cross_mask):
        B, T = 1, 1
        hs = self.decoder.embed_tokens(input_ids)

        for i, layer in enumerate(self.decoder.layers):
            # ── Self-attention ──
            res = hs
            h = layer.input_layernorm(hs)
            q = layer.self_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            k = layer.self_attn.k_proj(h).view(B, T, H, D).transpose(1, 2)
            v = layer.self_attn.v_proj(h).view(B, T, H, D).transpose(1, 2)

            qr = q[..., :ROT_DIM] * cos + rotate_half(q[..., :ROT_DIM]) * sin
            kr = k[..., :ROT_DIM] * cos + rotate_half(k[..., :ROT_DIM]) * sin
            q = torch.cat([qr, q[..., ROT_DIM:]], dim=-1)
            k = torch.cat([kr, k[..., ROT_DIM:]], dim=-1)

            sk = self.self_k
            sv = self.self_v
            w = write_onehot
            sk_new = sk[i] * (1 - w) + k * w
            sv_new = sv[i] * (1 - w) + v * w

            attn = q @ sk_new.transpose(-2, -1) * (D ** -0.5) + attn_mask
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ sv_new).transpose(1, 2).reshape(B, T, -1)
            hs = res + layer.self_attn.o_proj(out)
            self.self_k.data[i] = sk_new
            self.self_v.data[i] = sv_new

            # ── Cross-attention ──
            res = hs
            h = layer.post_attention_layernorm(hs)
            q = layer.encoder_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            ck = cross_k[i]
            cv = cross_v[i]
            cm = cross_mask
            attn = q @ ck.transpose(-2, -1) * (D ** -0.5) + cm
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ cv).transpose(1, 2).reshape(B, T, -1)
            hs = res + layer.encoder_attn.o_proj(out)

            # ── MLP ──
            res = hs
            h = layer.final_layernorm(hs)
            mlp = layer.mlp.fc1(h)
            up, gate = mlp.chunk(2, dim=-1)
            hs = res + layer.mlp.fc2(torch.nn.functional.silu(gate) * up)

        hs = self.decoder.norm(hs)
        return self.proj_out(hs)


def export():
    print("→ Exporting decoder (cross-KV as inputs, all-float32)…")
    model = DecoderWithCrossInputs().eval()

    args = (
        torch.tensor([[1]], dtype=torch.int32),
        torch.full((1, 1, 1, S_MAX), -1e4, dtype=torch.float32),
        torch.zeros(1, 1, 1, ROT_DIM, dtype=torch.float32),
        torch.zeros(1, 1, 1, ROT_DIM, dtype=torch.float32),
        torch.zeros(1, 1, S_MAX, 1, dtype=torch.float32),
        torch.zeros(NL, 1, H, S_ENC_MAX, D, dtype=torch.float32),
        torch.zeros(NL, 1, H, S_ENC_MAX, D, dtype=torch.float32),
        torch.zeros(1, 1, 1, S_ENC_MAX, dtype=torch.float32),
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
            ct.TensorType(name="cross_k",
                          shape=(NL, 1, H, S_ENC_MAX, D),
                          dtype=np.float32),
            ct.TensorType(name="cross_v",
                          shape=(NL, 1, H, S_ENC_MAX, D),
                          dtype=np.float32),
            ct.TensorType(name="cross_mask",
                          shape=(1, 1, 1, S_ENC_MAX),
                          dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        states=[
            ct.StateType(name="self_k",
                         wrapped_type=ct.TensorType(
                             shape=(NL, 1, H, S_MAX, D),
                             dtype=np.float32)),
            ct.StateType(name="self_v",
                         wrapped_type=ct.TensorType(
                             shape=(NL, 1, H, S_MAX, D),
                             dtype=np.float32)),
        ],
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18,
    )

    path = OUTPUT_DIR / "decoder_stateful.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    stateful.save(str(path))
    print(f"   Saved → {path}")

    # Verify
    mlmodel = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = mlmodel.get_spec()
    ins = {i.name: i for i in spec.description.input}
    states = {s.name for s in spec.description.state}
    assert "cross_k" in ins, "cross_k missing"
    assert "cross_v" in ins, "cross_v missing"
    assert "cross_mask" in ins, "cross_mask missing"
    assert "self_k" in states, "self_k not state"
    assert "self_v" in states, "self_v not state"
    print("   PASS: all-float32 decoder exported")


if __name__ == "__main__":
    export()
