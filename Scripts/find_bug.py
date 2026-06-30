#!/usr/bin/env python3
"""
Locate the bug by comparing replica vs HF intermediate values at step 1.

We've already established:
  - Step 0 matches HF.
  - Step 1 diverges.

Possible causes (in order of probability):
  1. RoPE: cos/sin reshape — replica uses shape [1,1,1,ROT_DIM] but RoPE in HF
     might multiply against [1,H,T,head_dim_split] differently.
  2. rotate_half: replica splits at ROT_DIM//2, HF might split at head_dim//2.
  3. Self-attention K/V order: replica stores RoPE'd K. HF stores raw K and
     rotates at query time.
  4. attn_mask scale: replica uses -1e4; HF may use -inf.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
from transformers.models.moonshine.modeling_moonshine import (
    MoonshineForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput

MODEL_NAME = "UsefulSensors/moonshine-tiny"
NL, H, D, HID = 6, 8, 36, 288
S_MAX, S_ENC_MAX, ROT_DIM = 128, 500, 32


def main():
    print("Loading HF…")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config._attn_implementation = "eager"
    hf = MoonshineForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)
    hf.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    BOS = hf.config.decoder_start_token_id

    print(f"\nModel config:")
    cfg = hf.config
    print(f"  num_attention_heads = {cfg.num_attention_heads}")
    print(f"  num_key_value_heads = {getattr(cfg, 'num_key_value_heads', '?')}")
    print(f"  head_dim            = {getattr(cfg, 'head_dim', '?')}")
    print(f"  hidden_size         = {cfg.hidden_size}")
    print(f"  decoder_layers      = {cfg.decoder_num_hidden_layers}")
    rt_factor = getattr(cfg, 'partial_rotary_factor', None)
    print(f"  partial_rotary_factor = {rt_factor}")
    if rt_factor is not None:
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        print(f"  → expected ROT_DIM = head_dim * partial_rotary_factor = {head_dim} * {rt_factor} = {head_dim * rt_factor}")

    # Check the rotate_half convention used by HF.
    import inspect
    from transformers.models.moonshine import modeling_moonshine
    src = inspect.getsource(modeling_moonshine)
    # find rotate_half
    if "def rotate_half" in src:
        idx = src.find("def rotate_half")
        snippet = src[idx:idx+500]
        print(f"\nHF rotate_half source:")
        for line in snippet.splitlines()[:12]:
            print(f"  {line}")

    # Find apply_rotary_pos_emb
    if "def apply_rotary_pos_emb" in src:
        idx = src.find("def apply_rotary_pos_emb")
        snippet = src[idx:idx+800]
        print(f"\nHF apply_rotary_pos_emb source:")
        for line in snippet.splitlines()[:16]:
            print(f"  {line}")

    # Show eager attention source (where RoPE is applied).
    if "class MoonshineAttention" in src:
        idx = src.find("class MoonshineAttention")
        snippet = src[idx:idx+2500]
        print(f"\nHF MoonshineAttention.forward (first part):")
        for line in snippet.splitlines()[:50]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
