#!/usr/bin/env python3
"""
Export Moonshine-base decoder WITHOUT state — cross-KV passed as inputs.
Adapted from export_nonstateful_decoder.py (streaming-small) for base model.
"""
import os, sys, json, shutil
import numpy as np
import torch
import coremltools as ct
from pathlib import Path
from transformers import AutoConfig
from transformers.models.moonshine.modeling_moonshine import MoonshineForConditionalGeneration

MODEL_NAME = "UsefulSensors/moonshine-base"
OUTPUT_DIR = Path.home() / ".cache" / "moonshine-coreml" / "base"

# base: hidden_size=416, heads=8, head_dim=52, layers=8, partial_rotary_factor=0.62
NL, H, D = 8, 8, 52
DEC_HID = 416
S_MAX, S_ENC_MAX, ROT_DIM = 128, 500, 32


def load_model():
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config._attn_implementation = "eager"
    model = MoonshineForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class NonStatefulDecoder(torch.nn.Module):
    """Single-step decoder that takes cross-KV as regular inputs (not state).

    This is slower than the stateful version (cross-KV passed every step)
    but avoids CoreML state API issues.
    """

    def __init__(self):
        super().__init__()
        model = load_model()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        for p in self.parameters():
            p.requires_grad = False
        # self-attn KV cache as regular buffers (updated in-place)
        self.register_buffer("self_k", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float16))
        self.register_buffer("self_v", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float16))
        self.eval()

    def forward(self, input_ids, attn_mask, cos, sin, write_onehot,
                cross_k, cross_v, cross_mask):
        B, T = 1, 1
        hs = self.decoder.embed_tokens(input_ids)

        sk_state = self.self_k.to(torch.float32)
        sv_state = self.self_v.to(torch.float32)
        ck_state = cross_k.to(torch.float32)
        cv_state = cross_v.to(torch.float32)
        cm_state = cross_mask.to(torch.float32)
        w = write_onehot.to(torch.float32)

        new_sk_layers = []
        new_sv_layers = []

        for i in range(NL):
            layer = self.decoder.layers[i]

            # Self-attention
            res = hs
            h = layer.input_layernorm(hs)
            q = layer.self_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            k = layer.self_attn.k_proj(h).view(B, T, H, D).transpose(1, 2)
            v = layer.self_attn.v_proj(h).view(B, T, H, D).transpose(1, 2)

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

            # Cross-attention
            res = hs
            h = layer.post_attention_layernorm(hs)
            q = layer.encoder_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            attn = q @ ck_state[i].transpose(-2, -1) * (D ** -0.5) + cm_state
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ cv_state[i]).transpose(1, 2).reshape(B, T, -1)
            hs = res + layer.encoder_attn.o_proj(out)

            # MLP (SwiGLU)
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


def export_nonstateful_decoder():
    print("-> Exporting non-stateful decoder for base (cross-KV as inputs)...")

    wrapper = NonStatefulDecoder().eval()

    # Trace inputs
    args = (
        torch.tensor([[1]], dtype=torch.int32),
        torch.full((1, 1, 1, S_MAX), -1e4, dtype=torch.float32).scatter_(
            -1, torch.zeros((1, 1, 1, 1), dtype=torch.long), 0.0),
        torch.zeros(1, 1, 1, ROT_DIM, dtype=torch.float32),
        torch.zeros(1, 1, 1, ROT_DIM, dtype=torch.float32),
        torch.zeros(1, 1, S_MAX, 1, dtype=torch.float32).scatter_(
            -1, torch.tensor([[[[0]]]]), 1.0),
        torch.zeros(NL, 1, H, S_ENC_MAX, D, dtype=torch.float16),
        torch.zeros(NL, 1, H, S_ENC_MAX, D, dtype=torch.float16),
        torch.full((1, 1, 1, S_ENC_MAX), -100.0, dtype=torch.float16),
    )
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, args)
    print("   JIT trace OK")

    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="attn_mask", shape=(1, 1, 1, S_MAX), dtype=np.float32),
            ct.TensorType(name="cos", shape=(1, 1, 1, ROT_DIM), dtype=np.float32),
            ct.TensorType(name="sin", shape=(1, 1, 1, ROT_DIM), dtype=np.float32),
            ct.TensorType(name="write_onehot", shape=(1, 1, S_MAX, 1), dtype=np.float32),
            ct.TensorType(name="cross_k", shape=(NL, 1, H, S_ENC_MAX, D), dtype=np.float16),
            ct.TensorType(name="cross_v", shape=(NL, 1, H, S_ENC_MAX, D), dtype=np.float16),
            ct.TensorType(name="cross_mask", shape=(1, 1, 1, S_ENC_MAX), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        states=[
            ct.StateType(name="self_k", wrapped_type=ct.TensorType(
                shape=(NL, 1, H, S_MAX, D), dtype=np.float16)),
            ct.StateType(name="self_v", wrapped_type=ct.TensorType(
                shape=(NL, 1, H, S_MAX, D), dtype=np.float16)),
        ],
        compute_units=ct.ComputeUnit.CPU_ONLY,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS18,
    )

    path = OUTPUT_DIR / "decoder_nonstateful.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    model.save(str(path))
    print(f"   Saved -> {path}")


if __name__ == "__main__":
    export_nonstateful_decoder()
