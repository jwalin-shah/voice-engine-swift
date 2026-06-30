#!/usr/bin/env python3
"""
Export Moonshine decoder with cross-KV as INPUTS (not state).

The exported model accepts:
  - input_ids (int32, [1,1]) — current token
  - attn_mask (float32, [1,1,1,128]) — causal self-attention mask
  - cos/sin (float32, [1,1,1,32]) — RoPE for current step
  - write_onehot (float32, [1,1,128,1]) — selects which position to write self-kv
  - cross_k (float16, [6,1,8,500,36]) — cross-attention keys per layer
  - cross_v (float16, [6,1,8,500,36]) — cross-attention values per layer
  - cross_mask (float16, [1,1,1,500]) — cross-attention mask

States: self_k, self_v (float16) — self-attention KV cache
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
    x1 = x[..., 0::2]; x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class Decoder(torch.nn.Module):
    """Cross-KV as inputs, self-KV as states. All-float32 compute."""
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config._attn_implementation = "eager"
        full = MoonshineForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)
        self.decoder = full.model.decoder
        self.proj_out = full.proj_out
        for p in self.parameters(): p.requires_grad = False
        # Self-KV states (fp16 — required by CoreML StateType)
        self.register_buffer("self_k", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float16))
        self.register_buffer("self_v", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float16))
        self.eval()

    def forward(self, input_ids, attn_mask, cos, sin, write_onehot,
                cross_k, cross_v, cross_mask):
        B, T = 1, 1
        hs = self.decoder.embed_tokens(input_ids)

        # Read state ONCE at start
        sk_state = self.self_k.to(torch.float32)
        sv_state = self.self_v.to(torch.float32)
        w = write_onehot

        new_sk_layers = []
        new_sv_layers = []

        for i, layer in enumerate(self.decoder.layers):
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

            # Cross-attention (from inputs, not state)
            res = hs
            h = layer.post_attention_layernorm(hs)
            q = layer.encoder_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            ck = cross_k[i].to(torch.float32)
            cv = cross_v[i].to(torch.float32)
            cm = cross_mask.to(torch.float32)
            attn = q @ ck.transpose(-2, -1) * (D ** -0.5) + cm
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ cv).transpose(1, 2).reshape(B, T, -1)
            hs = res + layer.encoder_attn.o_proj(out)

            # MLP
            res = hs
            h = layer.final_layernorm(hs)
            mlp = layer.mlp.fc1(h); up, gate = mlp.chunk(2, dim=-1)
            hs = res + layer.mlp.fc2(torch.nn.functional.silu(gate) * up)

        # Write ALL state at end (batched) — CoreML maps to single update_state op
        full_new_sk = torch.stack(new_sk_layers, dim=0).to(torch.float16)
        full_new_sv = torch.stack(new_sv_layers, dim=0).to(torch.float16)
        self.self_k[:] = full_new_sk
        self.self_v[:] = full_new_sv

        hs = self.decoder.norm(hs)
        return self.proj_out(hs)


def export():
    print("→ Exporting decoder (cross-KV as INPUTS, self-KV as STATES)…")
    model = Decoder().eval()

    traced = torch.jit.trace(model, (
        torch.tensor([[1]], dtype=torch.int32),
        torch.full((1, 1, 1, S_MAX), -1e4, dtype=torch.float32).scatter_(-1, torch.zeros((1,1,1,1), dtype=torch.long), 0),
        torch.zeros(1, 1, 1, ROT_DIM, dtype=torch.float32),
        torch.zeros(1, 1, 1, ROT_DIM, dtype=torch.float32),
        torch.zeros(1, 1, S_MAX, 1, dtype=torch.float32).scatter_(-1, torch.tensor([[[[0]]]]), 1),
        torch.zeros(NL, 1, H, S_ENC_MAX, D, dtype=torch.float16),
        torch.zeros(NL, 1, H, S_ENC_MAX, D, dtype=torch.float16),
        torch.zeros(1, 1, 1, S_ENC_MAX, dtype=torch.float16),
    ))
    print("   JIT trace OK")

    stateful = ct.convert(
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
            ct.StateType(name="self_k", wrapped_type=ct.TensorType(shape=(NL, 1, H, S_MAX, D), dtype=np.float16)),
            ct.StateType(name="self_v", wrapped_type=ct.TensorType(shape=(NL, 1, H, S_MAX, D), dtype=np.float16)),
        ],
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS18,
    )

    path = OUTPUT_DIR / "decoder_stateful.mlpackage"
    if path.exists(): shutil.rmtree(path)
    stateful.save(str(path))
    print(f"   Saved → {path}")

    # Verify
    mlmodel = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = mlmodel.get_spec()
    ins = {i.name for i in spec.description.input}
    sts = {s.name for s in spec.description.state}
    assert "cross_k" in ins, f"cross_k missing. Inputs: {sorted(ins)}"
    assert "cross_v" in ins
    assert "cross_mask" in ins
    assert "self_k" in sts
    assert "cross_k" not in sts
    print(f"   Inputs: {sorted(ins)}")
    print(f"   States: {sorted(sts)}")
    print("   PASS: cross-kv are INPUTS, self-kv are STATES")


if __name__ == "__main__":
    export()
