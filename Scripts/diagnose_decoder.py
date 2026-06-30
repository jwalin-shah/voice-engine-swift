#!/usr/bin/env python3
"""
Diagnose CoreML Moonshine decoder.

Goals:
  1. Determine the accepted dtype/shape for each state buffer.
  2. Test whether torch.jit.trace dropped the self_k/self_v in-place writes.
  3. Localize the root cause of garbled output.

Strategy: keep output short — single-line PASS/FAIL per check.
"""

import sys, time, contextlib, io
import numpy as np
from pathlib import Path

import coremltools as ct

MODEL_DIR = Path.home() / ".cache" / "moonshine-coreml" / "tiny-streaming"
NL, H, D, HID = 6, 8, 36, 288
S_MAX, S_ENC_MAX, ROT_DIM = 128, 500, 32


def try_write(state, name, arr):
    """Return (ok, err_msg). Suppress noisy stderr dumps from CoreML."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            state.write_state(name, arr)
        return True, None
    except Exception as e:
        msg = str(e).splitlines()[0][:80]
        return False, msg


def main():
    dec_path = MODEL_DIR / "decoder_stateful.mlpackage"
    if not dec_path.exists():
        print(f"FAIL: {dec_path} missing")
        sys.exit(1)

    print(f"Loading: {dec_path}")
    decoder = ct.models.MLModel(str(dec_path), compute_units=ct.ComputeUnit.CPU_ONLY)

    print("\n[1] State dtype compatibility")
    state_specs = [
        ("self_k",     (NL, 1, H, S_MAX, D)),
        ("self_v",     (NL, 1, H, S_MAX, D)),
        ("cross_k",    (NL, 1, H, S_ENC_MAX, D)),
        ("cross_v",    (NL, 1, H, S_ENC_MAX, D)),
        ("cross_mask", (1, 1, 1, S_ENC_MAX)),
    ]
    for name, shape in state_specs:
        state = decoder.make_state()
        for dt in (np.float32, np.float16):
            ok, err = try_write(state, name, np.zeros(shape, dtype=dt))
            status = "OK  " if ok else "FAIL"
            print(f"  {name:12s} shape={shape!s:30s} dt={dt.__name__:7s} → {status}"
                  + (f"  {err}" if err else ""))

    print("\n[2] Self-attention persistence (is jit.trace capturing the writes?)")

    # Build inputs that will produce *different* logits at step 1 depending on
    # whether step 0's K/V was stored. If self_k is never written, then
    # step 1 with attn_mask seeing positions 0+1 is identical to step 1 from
    # a fresh state (because position 0 has K=V=0 in both cases).

    state_specs_f32 = {n: (s, np.float32) for n, s in state_specs}

    def fresh_state(s_enc):
        state = decoder.make_state()
        ck = np.random.randn(NL, 1, H, S_ENC_MAX, D).astype(np.float32) * 0.1
        cv = np.random.randn(NL, 1, H, S_ENC_MAX, D).astype(np.float32) * 0.1
        cm = np.full((1, 1, 1, S_ENC_MAX), -1e4, dtype=np.float32)
        cm[..., :s_enc] = 0.0
        # zero out beyond s_enc
        ck[..., s_enc:, :] = 0
        cv[..., s_enc:, :] = 0
        for nm, arr in (("cross_k", ck), ("cross_v", cv), ("cross_mask", cm)):
            ok, err = try_write(state, nm, arr)
            if not ok:
                # try fp16
                ok, err = try_write(state, nm, arr.astype(np.float16))
                if not ok:
                    print(f"    cannot seed {nm}: {err}")
                    return None
        return state

    state_a = fresh_state(415)
    state_b = fresh_state(415)
    if state_a is None or state_b is None:
        print("  ABORT: cannot seed cross_* states")
        sys.exit(2)

    # Determine which fixed inputs the model expects.
    cos = np.zeros((1, 1, 1, ROT_DIM), dtype=np.float32)
    sin = np.zeros((1, 1, 1, ROT_DIM), dtype=np.float32)
    BOS, OTHER = 1, 99

    def step(state, token, pos):
        attn_mask = np.full((1, 1, 1, S_MAX), -1e4, dtype=np.float32)
        attn_mask[..., :pos+1] = 0.0
        onehot = np.zeros((1, 1, S_MAX, 1), dtype=np.float32)
        onehot[0, 0, pos, 0] = 1.0
        out = decoder.predict(
            {
                "input_ids": np.array([[token]], dtype=np.int32),
                "attn_mask": attn_mask,
                "cos": cos, "sin": sin,
                "write_onehot": onehot,
            },
            state=state,
        )
        return np.asarray(out["logits"])

    # state_a:  step 0 with BOS at pos 0  →  step 1 with OTHER at pos 1
    # state_b:                              →  step 1 with OTHER at pos 1 (no prior step 0)
    l0 = step(state_a, BOS, 0)
    print(f"  step 0 (state_a): logits range [{l0.min():.2f}, {l0.max():.2f}] argmax={int(l0[0,0].argmax())}")

    l1_warm = step(state_a, OTHER, 1)
    l1_cold = step(state_b, OTHER, 1)

    diff = np.abs(l1_warm - l1_cold)
    print(f"  step 1 warm vs cold:  max|Δ|={diff.max():.6f}  mean|Δ|={diff.mean():.6f}")
    print(f"     warm argmax={int(l1_warm[0,0].argmax())}  cold argmax={int(l1_cold[0,0].argmax())}")

    if diff.max() < 1e-3:
        print("\n  ❌ self_k IS NOT BEING WRITTEN by the CoreML decoder.")
        print("     Step 0's key/value did not influence step 1 — the trace")
        print("     dropped the in-place buffer writes.")
        print("     Fix: re-export using CoreML MIL Builder with explicit")
        print("     coreml_update_state ops, OR change the decoder forward")
        print("     to return new_k/new_v as outputs that map to states.")
    else:
        print("\n  ✓ self_k IS being written. Garbled-output bug is elsewhere")
        print("    (likely RoPE numerics or cross_mask).")

    print("\n[3] Cross-attention sanity (cross_mask shape)")
    # Run the same step with cross_mask all -1e4 vs all 0. Logits MUST differ.
    s1 = fresh_state(415)
    s2 = decoder.make_state()
    ck = np.zeros((NL,1,H,S_ENC_MAX,D), dtype=np.float32)
    cv = np.zeros((NL,1,H,S_ENC_MAX,D), dtype=np.float32)
    cm_off = np.full((1,1,1,S_ENC_MAX), -1e4, dtype=np.float32)
    for nm, arr in (("cross_k", ck), ("cross_v", cv), ("cross_mask", cm_off)):
        ok, err = try_write(s2, nm, arr) or try_write(s2, nm, arr.astype(np.float16))
    l_masked = step(s2, BOS, 0)
    l_unmasked = step(s1, BOS, 0)
    diff2 = np.abs(l_masked - l_unmasked).max()
    print(f"  fully-masked vs valid cross:  max|Δ|={diff2:.4f}")
    if diff2 < 1e-3:
        print("  ❌ Cross-attention has no effect → cross_k/v/mask aren't being read.")
    else:
        print("  ✓ Cross-attention is wired up correctly.")


if __name__ == "__main__":
    main()
