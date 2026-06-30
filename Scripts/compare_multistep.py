#!/usr/bin/env python3
"""
Multi-step parity: HF vs CoreML, generating 10 tokens greedily.

If single-step matches but multi-step diverges, the bug is in:
  - RoPE for positions > 0
  - attn_mask update logic
  - write_onehot indexing
  - self_k/v cache behavior across steps
"""
import warnings, contextlib, io
warnings.filterwarnings("ignore")

import numpy as np
import torch
from pathlib import Path
import coremltools as ct
from transformers import AutoConfig, AutoTokenizer
from transformers.models.moonshine.modeling_moonshine import MoonshineForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

MODEL_NAME = "UsefulSensors/moonshine-tiny"
MODEL_DIR = Path.home() / ".cache" / "moonshine-coreml" / "tiny-streaming"
NL, H, D, HID = 6, 8, 36, 288
S_MAX, S_ENC_MAX, ROT_DIM = 128, 500, 32
N_STEPS = 12


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

    # ── HF reference: greedy generate, capturing logits at each step ────────
    print("\n[A] HF greedy generation (no_cache mode for parity)")
    hf_tokens = [BOS]
    hf_logits_per_step = []
    with torch.no_grad():
        for step in range(N_STEPS):
            ids = torch.tensor([hf_tokens], dtype=torch.long)
            out = hf(
                decoder_input_ids=ids,
                encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits[0, -1].numpy()
            hf_logits_per_step.append(logits)
            next_tok = int(logits.argmax())
            hf_tokens.append(next_tok)
    print(f"  tokens: {hf_tokens}")
    print(f"  text:   '{tok.decode(hf_tokens, skip_special_tokens=True)}'")

    # ── Build cross_k/v ───────────────────────────────────────────────────
    print("\n[B] Building cross_k/v + RoPE tables")
    cross_k_list, cross_v_list = [], []
    for layer in hf.model.decoder.layers:
        k_w = layer.encoder_attn.k_proj.weight.detach().numpy().astype(np.float32)
        v_w = layer.encoder_attn.v_proj.weight.detach().numpy().astype(np.float32)
        eh = enc_hidden.numpy()
        k = eh @ k_w.T
        v = eh @ v_w.T
        if layer.encoder_attn.k_proj.bias is not None:
            k = k + layer.encoder_attn.k_proj.bias.detach().numpy().astype(np.float32)
        if layer.encoder_attn.v_proj.bias is not None:
            v = v + layer.encoder_attn.v_proj.bias.detach().numpy().astype(np.float32)
        k = k.reshape(1, S_enc_actual, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(1, S_enc_actual, H, D).transpose(0, 2, 1, 3)
        cross_k_list.append(k); cross_v_list.append(v)
    cross_k = np.stack(cross_k_list).astype(np.float32)
    cross_v = np.stack(cross_v_list).astype(np.float32)
    pad = S_ENC_MAX - S_enc_actual
    cross_k_p = np.pad(cross_k, ((0,0),(0,0),(0,0),(0,pad),(0,0))).astype(np.float32)
    cross_v_p = np.pad(cross_v, ((0,0),(0,0),(0,0),(0,pad),(0,0))).astype(np.float32)
    cross_mask = np.full((1,1,1,S_ENC_MAX), -1e4, dtype=np.float32)
    cross_mask[..., :S_enc_actual] = 0.0

    # RoPE for all positions 0..S_MAX-1
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
    # Apply HF's interleave transform so the tables match what the exported
    # decoder's (now-correct) rotate_half expects.
    half = cos_full.shape[-1] // 2
    cos_full = cos_full[..., :half].repeat_interleave(2, dim=-1)
    sin_full = sin_full[..., :half].repeat_interleave(2, dim=-1)
    cos_full = cos_full[0].numpy().astype(np.float32)
    sin_full = sin_full[0].numpy().astype(np.float32)
    print(f"  RoPE table shape: {cos_full.shape} (interleaved)")

    # ── CoreML greedy generation ──────────────────────────────────────────
    print(f"\n[C] CoreML greedy generation ({N_STEPS} steps)")
    dec = ct.models.MLModel(str(MODEL_DIR / "decoder_stateful.mlpackage"),
                            compute_units=ct.ComputeUnit.CPU_ONLY)
    state = dec.make_state()
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        state.write_state("cross_k", cross_k_p)
        state.write_state("cross_v", cross_v_p)
        state.write_state("cross_mask", cross_mask)

    cml_tokens = [BOS]
    cml_logits_per_step = []

    attn_mask = np.full((1,1,1,S_MAX), -1e4, dtype=np.float32)
    onehot = np.zeros((1,1,S_MAX,1), dtype=np.float32)

    for step in range(N_STEPS):
        attn_mask[..., step] = 0.0
        onehot[...] = 0.0
        onehot[0,0,step,0] = 1.0

        cos = cos_full[step].reshape(1,1,1,-1).astype(np.float32)
        sin = sin_full[step].reshape(1,1,1,-1).astype(np.float32)

        out = dec.predict({
            "input_ids": np.array([[cml_tokens[-1]]], dtype=np.int32),
            "attn_mask": attn_mask.copy(),
            "cos": cos,
            "sin": sin,
            "write_onehot": onehot.copy(),
        }, state=state)
        logits = np.asarray(out["logits"])[0, 0]
        cml_logits_per_step.append(logits)
        next_tok = int(logits.argmax())
        cml_tokens.append(next_tok)

    print(f"  tokens: {cml_tokens}")
    print(f"  text:   '{tok.decode(cml_tokens, skip_special_tokens=True)}'")

    # ── Step-by-step comparison ───────────────────────────────────────────
    print("\n[D] Per-step divergence")
    print(f"  {'step':>4} {'HF tok':>8} {'CML tok':>8} {'HF text':>15} {'CML text':>15} {'argmax':>7} {'|Δ| max':>9} {'corr':>8}")
    for i in range(N_STEPS):
        hf_t = int(hf_logits_per_step[i].argmax())
        cml_t = int(cml_logits_per_step[i].argmax())
        diff = np.abs(hf_logits_per_step[i] - cml_logits_per_step[i])
        hz = hf_logits_per_step[i]
        cz = cml_logits_per_step[i]
        hf_z = (hz - hz.mean()) / (hz.std() + 1e-9)
        cml_z = (cz - cz.mean()) / (cz.std() + 1e-9)
        corr = float((hf_z * cml_z).mean())
        match = "✓" if hf_t == cml_t else "✗"
        hf_text = tok.decode([hf_t]).replace('\n','\\n')[:15]
        cml_text = tok.decode([cml_t]).replace('\n','\\n')[:15]
        print(f"  {i:>4} {hf_t:>8} {cml_t:>8} {hf_text:>15} {cml_text:>15}  {match:>5}  {diff.max():>8.4f}  {corr:>7.4f}")

    # Find first divergence step.
    first_diverge = None
    for i in range(N_STEPS):
        if int(hf_logits_per_step[i].argmax()) != int(cml_logits_per_step[i].argmax()):
            first_diverge = i
            break

    print()
    if first_diverge is None:
        print("  ✓ ALL STEPS MATCH. CoreML is producing the same tokens as HF.")
    elif first_diverge == 0:
        print("  ❌ Diverges at step 0 — single-step bug (should not happen given earlier test)")
    else:
        print(f"  ⚠ First divergence at step {first_diverge}.")
        print(f"    Steps 0..{first_diverge-1} matched, then they split.")
        print(f"    Most likely: self_k cache accumulating wrong values.")
        print(f"    Less likely: RoPE for position {first_diverge}.")


if __name__ == "__main__":
    main()
