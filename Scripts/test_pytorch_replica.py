#!/usr/bin/env python3
"""
Test the EXACT same DecoderWithCrossMask logic in pure PyTorch (no CoreML).

If pure PyTorch with this logic produces good tokens, the bug is in the
CoreML conversion/trace.
If pure PyTorch also produces garbage, the bug is in the model logic itself
(blend, RoPE, mask format) — and fixing it in PyTorch will fix CoreML too.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
from transformers.models.moonshine.modeling_moonshine import MoonshineForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

MODEL_NAME = "UsefulSensors/moonshine-tiny"
NL, H, D, HID = 6, 8, 36, 288
S_MAX, S_ENC_MAX, ROT_DIM = 128, 500, 32
N_STEPS = 12


def rotate_half(x):
    # HF Moonshine uses INTERLEAVED rotation (GPT-J style), not split halves.
    x1 = x[..., 0::2]   # even indices
    x2 = x[..., 1::2]   # odd indices
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class DecoderReplica(torch.nn.Module):
    """Mirror of DecoderWithCrossMask in export_models.py."""
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config._attn_implementation = "eager"
        full = MoonshineForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)
        self.decoder = full.model.decoder
        self.proj_out = full.proj_out
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("cross_k", torch.zeros(NL, 1, H, S_ENC_MAX, D, dtype=torch.float16))
        self.register_buffer("cross_v", torch.zeros(NL, 1, H, S_ENC_MAX, D, dtype=torch.float16))
        self.register_buffer("cross_mask", torch.zeros(1, 1, 1, S_ENC_MAX, dtype=torch.float16))
        self.register_buffer("self_k", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float16))
        self.register_buffer("self_v", torch.zeros(NL, 1, H, S_MAX, D, dtype=torch.float16))
        self.eval()

    def forward(self, input_ids, attn_mask, cos, sin, write_onehot):
        B, T = 1, 1
        hs = self.decoder.embed_tokens(input_ids)

        for i, layer in enumerate(self.decoder.layers):
            res = hs
            h = layer.input_layernorm(hs)
            q = layer.self_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            k = layer.self_attn.k_proj(h).view(B, T, H, D).transpose(1, 2)
            v = layer.self_attn.v_proj(h).view(B, T, H, D).transpose(1, 2)
            qr = q[..., :ROT_DIM] * cos + rotate_half(q[..., :ROT_DIM]) * sin
            kr = k[..., :ROT_DIM] * cos + rotate_half(k[..., :ROT_DIM]) * sin
            q = torch.cat([qr, q[..., ROT_DIM:]], dim=-1)
            k = torch.cat([kr, k[..., ROT_DIM:]], dim=-1)

            sk16 = self.self_k.to(torch.float32)
            sv16 = self.self_v.to(torch.float32)
            w = write_onehot.to(torch.float32)
            sk_new = sk16[i] * (1 - w) + k * w
            sv_new = sv16[i] * (1 - w) + v * w

            attn = q @ sk_new.transpose(-2, -1) * (D ** -0.5) + attn_mask
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ sv_new).transpose(1, 2).reshape(B, T, -1)
            hs = res + layer.self_attn.o_proj(out)
            self.self_k.data[i] = sk_new.to(torch.float16)
            self.self_v.data[i] = sv_new.to(torch.float16)

            res = hs
            h = layer.post_attention_layernorm(hs)
            q = layer.encoder_attn.q_proj(h).view(B, T, H, D).transpose(1, 2)
            ck = self.cross_k.to(torch.float32)
            cv = self.cross_v.to(torch.float32)
            cm = self.cross_mask.to(torch.float32)
            attn = q @ ck[i].transpose(-2, -1) * (D ** -0.5) + cm
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ cv[i]).transpose(1, 2).reshape(B, T, -1)
            hs = res + layer.encoder_attn.o_proj(out)

            res = hs
            h = layer.final_layernorm(hs)
            mlp = layer.mlp.fc1(h)
            up, gate = mlp.chunk(2, dim=-1)
            hs = res + layer.mlp.fc2(torch.nn.functional.silu(gate) * up)

        hs = self.decoder.norm(hs)
        return self.proj_out(hs)


def main():
    print("Loading HF…")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config._attn_implementation = "eager"
    hf = MoonshineForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)
    hf.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    BOS = hf.config.decoder_start_token_id

    torch.manual_seed(42)
    S_enc_actual = 50
    enc_hidden = torch.randn(1, S_enc_actual, HID, dtype=torch.float32) * 0.5

    # ── HF reference ──────────────────────────────────────────────────────
    print("\n[A] HF reference")
    hf_tokens = [BOS]
    with torch.no_grad():
        for step in range(N_STEPS):
            out = hf(
                decoder_input_ids=torch.tensor([hf_tokens], dtype=torch.long),
                encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
                use_cache=False,
                return_dict=True,
            )
            hf_tokens.append(int(out.logits[0, -1].argmax()))
    print(f"  tokens: {hf_tokens}")
    print(f"  text:   '{tok.decode(hf_tokens, skip_special_tokens=True)}'")

    # ── Set up replica with cross_k/v from HF projections ─────────────────
    print("\n[B] PyTorch replica setup")
    replica = DecoderReplica().eval()

    cos_emb = hf.model.decoder.rotary_emb
    with torch.no_grad():
        try:
            cos_full, sin_full = cos_emb(
                torch.zeros(1, S_MAX, HID),
                position_ids=torch.arange(S_MAX).unsqueeze(0),
            )
        except TypeError:
            cos_full, sin_full = cos_emb(
                torch.zeros(1, S_MAX, HID),
                torch.arange(S_MAX).unsqueeze(0),
            )
    cos_full = cos_full[0]  # [S_MAX, ROT_DIM]
    sin_full = sin_full[0]

    # Seed cross_k/v and mask.
    with torch.no_grad():
        for i, layer in enumerate(hf.model.decoder.layers):
            k_w = layer.encoder_attn.k_proj.weight.float()
            v_w = layer.encoder_attn.v_proj.weight.float()
            k = enc_hidden @ k_w.T
            v = enc_hidden @ v_w.T
            if layer.encoder_attn.k_proj.bias is not None:
                k = k + layer.encoder_attn.k_proj.bias.float()
            if layer.encoder_attn.v_proj.bias is not None:
                v = v + layer.encoder_attn.v_proj.bias.float()
            k = k.view(1, S_enc_actual, H, D).transpose(1, 2)  # [1, H, S, D]
            v = v.view(1, S_enc_actual, H, D).transpose(1, 2)
            replica.cross_k.data[i, :, :, :S_enc_actual, :] = k.to(torch.float16)
            replica.cross_v.data[i, :, :, :S_enc_actual, :] = v.to(torch.float16)
        replica.cross_mask.data.fill_(-1e4)
        replica.cross_mask.data[..., :S_enc_actual] = 0.0

    # Apply the HF transform: take first half and repeat_interleave.
    # HF stores cos/sin at full ROT_DIM but only uses the first half, interleaved.
    half = cos_full.shape[-1] // 2
    cos_used = cos_full[..., :half].repeat_interleave(2, dim=-1)  # [S_MAX, ROT_DIM]
    sin_used = sin_full[..., :half].repeat_interleave(2, dim=-1)
    print(f"  Interleaved cos shape: {cos_used.shape}")

    # ── Run replica step by step ──────────────────────────────────────────
    print(f"\n[C] PyTorch replica greedy ({N_STEPS} steps)")
    replica_tokens = [BOS]
    with torch.no_grad():
        for step in range(N_STEPS):
            attn_mask = torch.full((1, 1, 1, S_MAX), -1e4, dtype=torch.float32)
            attn_mask[..., :step+1] = 0.0
            onehot = torch.zeros(1, 1, S_MAX, 1, dtype=torch.float32)
            onehot[0, 0, step, 0] = 1.0
            cos = cos_used[step].view(1, 1, 1, -1).to(torch.float32)
            sin = sin_used[step].view(1, 1, 1, -1).to(torch.float32)
            input_ids = torch.tensor([[replica_tokens[-1]]], dtype=torch.int32)

            logits = replica(input_ids, attn_mask, cos, sin, onehot)
            next_tok = int(logits[0, -1].argmax())
            replica_tokens.append(next_tok)

    print(f"  tokens: {replica_tokens}")
    print(f"  text:   '{tok.decode(replica_tokens, skip_special_tokens=True)}'")

    print("\n[D] Verdict")
    match = (hf_tokens == replica_tokens)
    if match:
        print("  ✓ Pure PyTorch replica matches HF token-for-token.")
        print("    → The model logic is correct.")
        print("    → The bug is in the CoreML conversion / trace.")
        print("    → Next step: re-export with simpler ops or use MIL Builder.")
    else:
        first_diff = next((i for i in range(len(hf_tokens)) if hf_tokens[i] != replica_tokens[i]), None)
        print(f"  ✗ Pure PyTorch replica diverges from HF at step {first_diff}.")
        print(f"    → The model LOGIC is wrong — broadcasting, RoPE, or mask format.")
        print(f"    → CoreML faithfully reproduces a broken model.")
        print(f"    → Fix the PyTorch version first; CoreML will follow.")


if __name__ == "__main__":
    main()
