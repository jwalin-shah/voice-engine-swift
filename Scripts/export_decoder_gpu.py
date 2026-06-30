#!/usr/bin/env python3
"""
Export decoder with CPU_AND_GPU compute to test GPU acceleration.

Usage:
  ! python3 voice-engine-swift/Scripts/export_decoder_gpu.py
  ! python3 voice-engine-swift/bench.py --iterations 10 --json
  # Compare with:
  ! python3 voice-engine-swift/bench.py --iterations 10 --json --duration 10
"""
import os, sys, shutil
import numpy as np, torch
import coremltools as ct
from pathlib import Path

MODEL_NAME = "UsefulSensors/moonshine-tiny"
OUTPUT_DIR = Path.home() / ".cache" / "moonshine-coreml" / "tiny-streaming"
BACKUP = Path.home() / ".cache" / "moonshine-coreml" / "tiny-streaming-cpu-only"

NL, H, D, HID = 6, 8, 36, 288
S_MAX, S_ENC_MAX, ROT_DIM = 128, 500, 32

def rotate_half(x):
    x1 = x[..., 0::2]; x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class DecoderWithCrossInputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoConfig
        from transformers.models.moonshine.modeling_moonshine import MoonshineForConditionalGeneration
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config._attn_implementation = "eager"
        full = MoonshineForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)
        self.decoder = full.model.decoder; self.proj_out = full.proj_out
        for p in self.parameters(): p.requires_grad = False
        self.register_buffer("self_k", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float16))
        self.register_buffer("self_v", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float16))
        self.eval()

    def forward(self, input_ids, attn_mask, cos, sin, write_onehot, cross_k, cross_v, cross_mask):
        B, T = 1, 1
        hs = self.decoder.embed_tokens(input_ids)
        for i, layer in enumerate(self.decoder.layers):
            res = hs; h = layer.input_layernorm(hs)
            q = layer.self_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            k = layer.self_attn.k_proj(h).view(B, T, H, D).transpose(1, 2)
            v = layer.self_attn.v_proj(h).view(B, T, H, D).transpose(1, 2)
            qr = q[..., :ROT_DIM] * cos + rotate_half(q[..., :ROT_DIM]) * sin
            kr = k[..., :ROT_DIM] * cos + rotate_half(k[..., :ROT_DIM]) * sin
            q = torch.cat([qr, q[..., ROT_DIM:]], dim=-1)
            k = torch.cat([kr, k[..., ROT_DIM:]], dim=-1)
            sk16 = self.self_k.to(torch.float32); sv16 = self.self_v.to(torch.float32)
            w = write_onehot.to(torch.float32)
            sk_new = sk16[i] * (1 - w) + k * w; sv_new = sv16[i] * (1 - w) + v * w
            attn = q @ sk_new.transpose(-2, -1) * (D ** -0.5) + attn_mask
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ sv_new).transpose(1, 2).reshape(B, T, -1)
            hs = res + layer.self_attn.o_proj(out)
            self.self_k.data[i] = sk_new.to(torch.float16); self.self_v.data[i] = sv_new.to(torch.float16)
            res = hs; h = layer.post_attention_layernorm(hs)
            q = layer.encoder_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            ck = cross_k[i].to(torch.float32); cv = cross_v[i].to(torch.float32)
            cm = cross_mask.to(torch.float32)
            attn = q @ ck.transpose(-2, -1) * (D ** -0.5) + cm
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ cv).transpose(1, 2).reshape(B, T, -1)
            hs = res + layer.encoder_attn.o_proj(out)
            res = hs; h = layer.final_layernorm(hs)
            mlp = layer.mlp.fc1(h); up, gate = mlp.chunk(2, dim=-1)
            hs = res + layer.mlp.fc2(torch.nn.functional.silu(gate) * up)
        hs = self.decoder.norm(hs)
        return self.proj_out(hs)

def export():
    print("→ Backing up current CPU_ONLY decoder...")
    dec_path = OUTPUT_DIR / "decoder_stateful.mlpackage"
    backup_path = BACKUP / "decoder_stateful.mlpackage"
    if dec_path.exists():
        BACKUP.mkdir(parents=True, exist_ok=True)
        if backup_path.exists(): shutil.rmtree(backup_path)
        shutil.copytree(dec_path, backup_path)
        print(f"   Backed up → {backup_path}")

    print("→ Exporting decoder with CPU_AND_GPU...")
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
    with torch.no_grad(): traced = torch.jit.trace(model, args)
    print("   JIT trace OK")

    # KEY CHANGE: CPU_AND_GPU instead of CPU_ONLY
    stateful = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="attn_mask", shape=(1, 1, 1, S_MAX), dtype=np.float32),
            ct.TensorType(name="cos", shape=(1, 1, 1, ROT_DIM), dtype=np.float32),
            ct.TensorType(name="sin", shape=(1, 1, 1, ROT_DIM), dtype=np.float32),
            ct.TensorType(name="write_onehot", shape=(1, 1, S_MAX, 1), dtype=np.float32),
            ct.TensorType(name="cross_k", shape=(NL, 1, H, S_ENC_MAX, D), dtype=np.float32),
            ct.TensorType(name="cross_v", shape=(NL, 1, H, S_ENC_MAX, D), dtype=np.float32),
            ct.TensorType(name="cross_mask", shape=(1, 1, 1, S_ENC_MAX), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        states=[
            ct.StateType(name="self_k", wrapped_type=ct.TensorType(shape=(NL, 1, H, S_MAX, D), dtype=np.float16)),
            ct.StateType(name="self_v", wrapped_type=ct.TensorType(shape=(NL, 1, H, S_MAX, D), dtype=np.float16)),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,  # <-- GPU!
        minimum_deployment_target=ct.target.iOS18,
    )

    if dec_path.exists(): shutil.rmtree(dec_path)
    stateful.save(str(dec_path))
    print(f"   Saved → {dec_path}")
    print("\n✓ GPU decoder ready. Run bench:")
    print("  python3 voice-engine-swift/bench.py --iterations 10 --json")
    print("\nTo restore CPU_ONLY:")
    print("  rm -rf ~/.cache/moonshine-coreml/tiny-streaming/decoder_stateful.mlpackage")
    print("  cp -R ~/.cache/moonshine-coreml/tiny-streaming-cpu-only/decoder_stateful.mlpackage \\")
    print("    ~/.cache/moonshine-coreml/tiny-streaming/decoder_stateful.mlpackage")

if __name__ == "__main__":
    export()
