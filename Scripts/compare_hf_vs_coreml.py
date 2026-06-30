#!/usr/bin/env python3
"""
Numerical parity test: HF decoder vs CoreML decoder, single step.

Strategy:
  1. Load HF Moonshine. Capture its full forward(input_ids=[BOS], encoder_hidden=H)
     to get reference logits.
  2. Run the CoreML decoder with the same inputs (BOS, position 0, same
     encoder hidden states fed via cross_k/cross_v from HF's projections).
  3. Compare logits.
  4. If they diverge significantly, walk earlier in the graph: compare cross_k,
     cross_v, RoPE, attention scores.
"""
import warnings, contextlib, io
warnings.filterwarnings("ignore")

import sys
import numpy as np
import torch
from pathlib import Path
import coremltools as ct
from transformers import AutoConfig, AutoTokenizer
from transformers.models.moonshine.modeling_moonshine import MoonshineForConditionalGeneration

MODEL_NAME = "UsefulSensors/moonshine-tiny"
MODEL_DIR = Path.home() / ".cache" / "moonshine-coreml" / "tiny-streaming"
NL, H, D, HID = 6, 8, 36, 288
S_MAX, S_ENC_MAX, ROT_DIM = 128, 500, 32


def main():
    print("Loading HF model…")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config._attn_implementation = "eager"
    hf = MoonshineForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)
    hf.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"BOS={tok.bos_token_id}  EOS={tok.eos_token_id}  decoder_start={hf.config.decoder_start_token_id}")
    BOS = hf.config.decoder_start_token_id or tok.bos_token_id or 1

    # Build a fake encoder hidden state — random but realistic magnitude.
    # We'll use the SAME tensor for both HF and CoreML so we factor out the
    # encoder export entirely.
    S_enc_actual = 50  # short sequence
    torch.manual_seed(42)
    enc_hidden = torch.randn(1, S_enc_actual, HID, dtype=torch.float32) * 0.5

    # ── HF forward: single decoder step with BOS ────────────────────────────
    print("\n[A] HF reference forward")
    from transformers.modeling_outputs import BaseModelOutput
    with torch.no_grad():
        hf_out = hf(
            decoder_input_ids=torch.tensor([[BOS]], dtype=torch.long),
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
            use_cache=False,
            return_dict=True,
        )
    hf_logits = hf_out.logits[0, 0].numpy()  # [vocab]
    hf_argmax = int(hf_logits.argmax())
    print(f"  argmax={hf_argmax}  top tok='{tok.decode([hf_argmax])}'  range=[{hf_logits.min():.2f},{hf_logits.max():.2f}]")

    # ── Compute cross_k, cross_v exactly as bench.py does ────────────────────
    print("\n[B] Compute cross_k/v from HF weights")
    cross_k_list, cross_v_list = [], []
    for layer in hf.model.decoder.layers:
        k_w = layer.encoder_attn.k_proj.weight.detach().numpy().astype(np.float32)
        v_w = layer.encoder_attn.v_proj.weight.detach().numpy().astype(np.float32)
        k_b = layer.encoder_attn.k_proj.bias
        v_b = layer.encoder_attn.v_proj.bias
        eh = enc_hidden.numpy()
        k = eh @ k_w.T
        v = eh @ v_w.T
        if k_b is not None:
            k = k + k_b.detach().numpy().astype(np.float32)
        if v_b is not None:
            v = v + v_b.detach().numpy().astype(np.float32)
        # Reshape [1, S, HID] → [1, S, H, D] → [1, H, S, D]
        k = k.reshape(1, S_enc_actual, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(1, S_enc_actual, H, D).transpose(0, 2, 1, 3)
        cross_k_list.append(k)
        cross_v_list.append(v)
    cross_k = np.stack(cross_k_list).astype(np.float32)  # [NL, 1, H, S_enc, D]
    cross_v = np.stack(cross_v_list).astype(np.float32)
    print(f"  cross_k shape={cross_k.shape} range=[{cross_k.min():.2f},{cross_k.max():.2f}]")

    # Pad to S_ENC_MAX
    pad = S_ENC_MAX - S_enc_actual
    cross_k_padded = np.pad(cross_k, ((0,0),(0,0),(0,0),(0,pad),(0,0))).astype(np.float32)
    cross_v_padded = np.pad(cross_v, ((0,0),(0,0),(0,0),(0,pad),(0,0))).astype(np.float32)
    cross_mask = np.full((1,1,1,S_ENC_MAX), -1e4, dtype=np.float32)
    cross_mask[..., :S_enc_actual] = 0.0

    # ── Compute RoPE cos/sin for position 0 ──────────────────────────────────
    cos_emb = hf.model.decoder.rotary_emb
    pos_ids = torch.arange(S_MAX)
    with torch.no_grad():
        # MoonshineRotaryEmbedding returns (cos, sin) for given positions.
        # API varies across transformers versions.
        try:
            cos_full, sin_full = cos_emb(torch.zeros(1, S_MAX, HID), position_ids=pos_ids.unsqueeze(0))
        except TypeError:
            cos_full, sin_full = cos_emb(torch.zeros(1, S_MAX, HID), pos_ids.unsqueeze(0))
    cos_full = cos_full[0].numpy().astype(np.float32)  # [S_MAX, ROT_DIM]
    sin_full = sin_full[0].numpy().astype(np.float32)
    print(f"  RoPE cos[0]={cos_full[0,:4]}  shape={cos_full.shape}")

    cos0 = cos_full[0].reshape(1, 1, 1, -1).astype(np.float32)
    sin0 = sin_full[0].reshape(1, 1, 1, -1).astype(np.float32)

    # ── CoreML decoder forward, step 0 ────────────────────────────────────────
    print("\n[C] CoreML decoder step 0")
    dec = ct.models.MLModel(str(MODEL_DIR / "decoder_stateful.mlpackage"),
                            compute_units=ct.ComputeUnit.CPU_ONLY)
    state = dec.make_state()

    # Suppress noisy stderr from write_state errors
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        state.write_state("cross_k", cross_k_padded)
        state.write_state("cross_v", cross_v_padded)
        state.write_state("cross_mask", cross_mask)

    attn_mask = np.full((1,1,1,S_MAX), -1e4, dtype=np.float32)
    attn_mask[..., 0] = 0.0
    onehot = np.zeros((1,1,S_MAX,1), dtype=np.float32)
    onehot[0,0,0,0] = 1.0

    out = dec.predict({
        "input_ids": np.array([[BOS]], dtype=np.int32),
        "attn_mask": attn_mask,
        "cos": cos0,
        "sin": sin0,
        "write_onehot": onehot,
    }, state=state)
    cml_logits = np.asarray(out["logits"])[0, 0]
    cml_argmax = int(cml_logits.argmax())
    print(f"  argmax={cml_argmax}  top tok='{tok.decode([cml_argmax])}'  range=[{cml_logits.min():.2f},{cml_logits.max():.2f}]")

    # ── Compare ───────────────────────────────────────────────────────────────
    print("\n[D] HF vs CoreML logit comparison")
    diff = np.abs(hf_logits - cml_logits)
    print(f"  |Δ| max  = {diff.max():.4f}")
    print(f"  |Δ| mean = {diff.mean():.4f}")
    print(f"  |Δ| std  = {diff.std():.4f}")

    # Top-5 from each
    hf_top5 = hf_logits.argsort()[-5:][::-1]
    cml_top5 = cml_logits.argsort()[-5:][::-1]
    print(f"\n  HF   top-5: {hf_top5.tolist()}  → '{tok.decode(hf_top5.tolist())}'")
    print(f"  CML  top-5: {cml_top5.tolist()}  → '{tok.decode(cml_top5.tolist())}'")

    # Check correlation (logit shape, ignoring offset)
    hf_z = (hf_logits - hf_logits.mean()) / (hf_logits.std() + 1e-9)
    cml_z = (cml_logits - cml_logits.mean()) / (cml_logits.std() + 1e-9)
    corr = float((hf_z * cml_z).mean())
    print(f"\n  Pearson correlation = {corr:.6f}")

    if corr > 0.999:
        print("\n  ✓ CoreML decoder matches HF within float precision.")
        print("    The garbled-output bug must be in either:")
        print("      (a) the CoreML ENCODER export (different hidden states than HF),")
        print("      (b) the cross_k/v projection in bench.py (already matches HF here),")
        print("      (c) the position/RoPE indexing in the multi-step loop.")
    elif corr > 0.9:
        print("\n  ⚠ Close but not exact. Likely RoPE precision or fp16 in CoreML.")
    else:
        print("\n  ❌ Significant divergence. CoreML decoder is computing something different.")
        print("     Likely culprits in priority order:")
        print("       1. RoPE: the export inlined RoPE tables for a different sin/cos")
        print("          convention than HF's runtime computation.")
        print("       2. attn_mask format: tracer froze a specific mask shape.")
        print("       3. write_onehot: tracer captured a specific position behavior.")


if __name__ == "__main__":
    main()
