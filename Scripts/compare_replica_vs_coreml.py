#!/usr/bin/env python3
"""
Run BOTH the pure-PyTorch replica (proven to match HF) AND the re-exported
CoreML decoder with IDENTICAL inputs at each step. They should produce
matching logits if the CoreML export preserved the corrected logic.

If they diverge: the trace/conversion is still dropping something.
If they match: the bug we saw in compare_multistep was its own bug,
not the export.
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

import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_pytorch_replica import DecoderReplica, rotate_half  # uses fixed rotate_half

MODEL_NAME = "UsefulSensors/moonshine-tiny"
MODEL_DIR = Path.home() / ".cache" / "moonshine-coreml" / "tiny-streaming"
NL, H, D, HID = 6, 8, 36, 288
S_MAX, S_ENC_MAX, ROT_DIM = 128, 500, 32
N_STEPS = 8


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

    # ── Build inputs shared between replica and CoreML ──────────────────────
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
    # HF interleave transform
    half = cos_full.shape[-1] // 2
    cos_used = cos_full[..., :half].repeat_interleave(2, dim=-1)
    sin_used = sin_full[..., :half].repeat_interleave(2, dim=-1)

    # Build cross_k/v
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
        cross_k_list.append(k)
        cross_v_list.append(v)
    cross_k = np.stack(cross_k_list).astype(np.float32)
    cross_v = np.stack(cross_v_list).astype(np.float32)
    pad = S_ENC_MAX - S_enc_actual
    cross_k_p = np.pad(cross_k, ((0,0),(0,0),(0,0),(0,pad),(0,0))).astype(np.float32)
    cross_v_p = np.pad(cross_v, ((0,0),(0,0),(0,0),(0,pad),(0,0))).astype(np.float32)
    cross_mask = np.full((1,1,1,S_ENC_MAX), -1e4, dtype=np.float32)
    cross_mask[..., :S_enc_actual] = 0.0

    # ── Setup PyTorch replica ──────────────────────────────────────────────
    print("Setting up replica…")
    replica = DecoderReplica().eval()
    with torch.no_grad():
        # Same cross_k/v seeding
        for i, layer in enumerate(hf.model.decoder.layers):
            k_w = layer.encoder_attn.k_proj.weight.float()
            v_w = layer.encoder_attn.v_proj.weight.float()
            k = enc_hidden @ k_w.T
            v = enc_hidden @ v_w.T
            if layer.encoder_attn.k_proj.bias is not None:
                k = k + layer.encoder_attn.k_proj.bias.float()
            if layer.encoder_attn.v_proj.bias is not None:
                v = v + layer.encoder_attn.v_proj.bias.float()
            k = k.view(1, S_enc_actual, H, D).transpose(1, 2)
            v = v.view(1, S_enc_actual, H, D).transpose(1, 2)
            replica.cross_k.data[i, :, :, :S_enc_actual, :] = k.to(torch.float16)
            replica.cross_v.data[i, :, :, :S_enc_actual, :] = v.to(torch.float16)
        replica.cross_mask.data.fill_(-1e4)
        replica.cross_mask.data[..., :S_enc_actual] = 0.0

    # ── Setup CoreML ────────────────────────────────────────────────────────
    print("Loading CoreML decoder…")
    dec = ct.models.MLModel(str(MODEL_DIR / "decoder_stateful.mlpackage"),
                            compute_units=ct.ComputeUnit.CPU_ONLY)
    state = dec.make_state()
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        state.write_state("cross_k", cross_k_p)
        state.write_state("cross_v", cross_v_p)
        state.write_state("cross_mask", cross_mask)

    # ── Run side-by-side ────────────────────────────────────────────────────
    print(f"\nRunning {N_STEPS} steps — same inputs, compare logits\n")
    print(f"  {'step':>4} {'rep tok':>8} {'cml tok':>8} {'rep text':>12} {'cml text':>12} {'argmax':>7} {'|Δ| max':>9} {'corr':>8}")

    rep_tokens = [BOS]
    cml_tokens = [BOS]

    for step in range(N_STEPS):
        # Build common inputs
        attn_mask_np = np.full((1, 1, 1, S_MAX), -1e4, dtype=np.float32)
        attn_mask_np[..., :step+1] = 0.0
        onehot_np = np.zeros((1, 1, S_MAX, 1), dtype=np.float32)
        onehot_np[0, 0, step, 0] = 1.0
        cos_np = cos_used[0, step].numpy().reshape(1, 1, 1, -1).astype(np.float32)
        sin_np = sin_used[0, step].numpy().reshape(1, 1, 1, -1).astype(np.float32)

        # Replica
        with torch.no_grad():
            rep_logits = replica(
                torch.tensor([[rep_tokens[-1]]], dtype=torch.int32),
                torch.from_numpy(attn_mask_np),
                torch.from_numpy(cos_np),
                torch.from_numpy(sin_np),
                torch.from_numpy(onehot_np),
            )[0, -1].numpy()
        rep_tok = int(rep_logits.argmax())
        rep_tokens.append(rep_tok)

        # CoreML — feed the SAME token that replica used (force both onto same path)
        cml_out = dec.predict({
            "input_ids": np.array([[rep_tokens[-2]]], dtype=np.int32),  # use replica's prior token
            "attn_mask": attn_mask_np.copy(),
            "cos": cos_np,
            "sin": sin_np,
            "write_onehot": onehot_np.copy(),
        }, state=state)
        cml_logits = np.asarray(cml_out["logits"])[0, 0]
        cml_tok = int(cml_logits.argmax())
        cml_tokens.append(cml_tok)

        diff = np.abs(rep_logits - cml_logits)
        rz = (rep_logits - rep_logits.mean()) / (rep_logits.std() + 1e-9)
        cz = (cml_logits - cml_logits.mean()) / (cml_logits.std() + 1e-9)
        corr = float((rz * cz).mean())
        match = "✓" if rep_tok == cml_tok else "✗"
        rt = tok.decode([rep_tok]).replace("\n","\\n")[:12]
        ct_ = tok.decode([cml_tok]).replace("\n","\\n")[:12]
        print(f"  {step:>4} {rep_tok:>8} {cml_tok:>8} {rt:>12} {ct_:>12}  {match:>5}  {diff.max():>8.4f}  {corr:>7.4f}")


if __name__ == "__main__":
    main()
