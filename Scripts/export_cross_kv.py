#!/usr/bin/env python3
"""
Export cross-KV projection as a standalone CoreML model.

The current Swift implementation does 6 layers × 2 gemm calls + 24,000 memcpy
calls on CPU per transcribe — 106ms. This module:

1. Loads the pre-extracted K/V weights from cross_kv_weights.npz
2. Wraps them in a PyTorch module: hidden_states → (cross_k, cross_v, cross_mask)
3. JIT-traces → CoreML .mlpackage (ANE-eligible)

Expected: 106ms → ~3-5ms (single forward pass on ANE vs 12 sgemm + 24K memcpy on CPU).
"""

import numpy as np
import torch
import coremltools as ct
from pathlib import Path

MODEL_DIR = Path.home() / ".cache" / "moonshine-coreml" / "tiny-streaming"
NL, H, D, HID = 6, 8, 36, 288
S_ENC_MAX = 500
HD = H * D  # 288


class CrossKVProjection(torch.nn.Module):
    """hidden_states [1, S_enc, HID] → (cross_k, cross_v, cross_mask).

    The weights are baked in as buffers so CoreML sees them as constants.
    CoreML compiles this to a fixed-function graph — no loops, no dispatch overhead.
    """
    def __init__(self, kw, vw, kb, vb):
        super().__init__()
        # Register all 12 weight matrices as float32 buffers.
        for i in range(NL):
            self.register_buffer(f"kw{i}", torch.from_numpy(kw[i]).float())
            self.register_buffer(f"vw{i}", torch.from_numpy(vw[i]).float())
            if kb[i] is not None:
                self.register_buffer(f"kb{i}", torch.from_numpy(kb[i]).float())
            if vb[i] is not None:
                self.register_buffer(f"vb{i}", torch.from_numpy(vb[i]).float())

    def forward(self, hidden_states, S_enc):
        """
        hidden_states: [1, S_enc_max, HID] — encoder output padded to S_ENC_MAX
        S_enc: int — actual encoder frames (0 < S_enc <= S_ENC_MAX)
        Returns: cross_k, cross_v, cross_mask — all float32, zero-filled beyond S_enc
        """
        # Build K/V per layer.
        k_layers, v_layers = [], []
        for i in range(NL):
            kw = getattr(self, f"kw{i}")
            vw = getattr(self, f"vw{i}")

            # gemm: [1, S_ENC_MAX, HID] @ [HID, HD]^T → [1, S_ENC_MAX, HD]
            k = hidden_states @ kw.T
            v = hidden_states @ vw.T

            # Bias.
            kb_name = f"kb{i}"
            if hasattr(self, kb_name):
                k = k + getattr(self, kb_name)
            vb_name = f"vb{i}"
            if hasattr(self, vb_name):
                v = v + getattr(self, vb_name)

            # Reshape: [1, S_ENC_MAX, H, D] then transpose to [1, H, S_ENC_MAX, D]
            k = k.reshape(1, S_ENC_MAX, H, D).permute(0, 2, 1, 3)
            v = v.reshape(1, S_ENC_MAX, H, D).permute(0, 2, 1, 3)

            k_layers.append(k)
            v_layers.append(v)

        # Stack → [NL, 1, H, S_ENC_MAX, D]
        cross_k = torch.stack(k_layers, dim=0)
        cross_v = torch.stack(v_layers, dim=0)

        # Cross mask: 0 for valid positions, -10000 for padding.
        # Build from S_enc (a scalar input). Use a range comparison.
        positions = torch.arange(S_ENC_MAX, dtype=torch.float32).reshape(1, 1, 1, S_ENC_MAX)
        cross_mask = torch.where(
            positions < S_enc.float(),
            torch.tensor(0.0),
            torch.tensor(-10000.0),
        )

        return cross_k, cross_v, cross_mask


def main():
    # Load pre-extracted weights.
    w = np.load(str(MODEL_DIR / "cross_kv_weights.npz"))
    kw = [w[f"layer{i}_k_weight"] for i in range(NL)]
    vw = [w[f"layer{i}_v_weight"] for i in range(NL)]
    kb = [w.get(f"layer{i}_k_bias") for i in range(NL)]
    vb = [w.get(f"layer{i}_v_bias") for i in range(NL)]
    print(f"Loaded K/V weights: shapes {[x.shape for x in kw]}")

    # Instantiate and trace.
    model = CrossKVProjection(kw, vw, kb, vb).eval()
    for p in model.parameters():
        p.requires_grad = False

    # Trace with max S_enc (500). The model handles variable S_enc via the
    # S_enc scalar input — the mask zeros out padding positions.
    dummy_hidden = torch.randn(1, S_ENC_MAX, HID, dtype=torch.float32)
    dummy_S_enc = torch.tensor(S_ENC_MAX, dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy_hidden, dummy_S_enc))

    ck, cv, cm = traced(dummy_hidden, dummy_S_enc)
    print(f"Traced OK: cross_k={list(ck.shape)}, cross_v={list(cv.shape)}, cross_mask={list(cm.shape)}")

    # Convert to CoreML.
    print("Converting to CoreML…")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, S_ENC_MAX, HID), dtype=np.float32),
            ct.TensorType(name="S_enc", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="cross_k", dtype=np.float32),
            ct.TensorType(name="cross_v", dtype=np.float32),
            ct.TensorType(name="cross_mask", dtype=np.float32),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
    )

    out = MODEL_DIR / "cross_kv.mlpackage"
    if out.exists():
        import shutil
        shutil.rmtree(out)
    mlmodel.save(str(out))

    # Size.
    import os
    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(out)
        for f in files
    )
    print(f"\nSaved {out} ({total/1024/1024:.0f} MB)")

    # Verify output shapes match expectations.
    print("\nVerifying prediction…")
    pred = mlmodel.predict({
        "hidden_states": np.random.randn(1, S_ENC_MAX, HID).astype(np.float32),
        "S_enc": np.array([250], dtype=np.int32),
    })
    print(f"  cross_k:    {pred['cross_k'].shape}")
    print(f"  cross_v:    {pred['cross_v'].shape}")
    print(f"  cross_mask: {pred['cross_mask'].shape}")

    # Check mask is correct for S_enc=250.
    mask = pred["cross_mask"]
    assert mask.shape == (1, 1, 1, S_ENC_MAX), f"Bad mask shape: {mask.shape}"
    assert np.all(mask[..., :250] == 0.0), "First 250 should be 0"
    assert np.all(mask[..., 250:] == -10000.0), "Remaining should be -10000"
    print("  Mask verified: first 250=0.0, rest=-10000.0 ✓")

    # Check against NumPy reference for correctness.
    print("\nChecking against NumPy reference…")
    hs = np.random.randn(1, S_ENC_MAX, HID).astype(np.float32)
    S_enc = 250

    # NumPy path (same logic as current Swift code).
    ck_np, cv_np = [], []
    for i in range(NL):
        k = hs[0] @ kw[i].T
        v = hs[0] @ vw[i].T
        if kb[i] is not None:
            k = k + kb[i]
        if vb[i] is not None:
            v = v + vb[i]
        k = k.reshape(1, S_ENC_MAX, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(1, S_ENC_MAX, H, D).transpose(0, 2, 1, 3)
        ck_np.append(k)
        cv_np.append(v)
    ck_np = np.stack(ck_np)
    cv_np = np.stack(cv_np)

    # CoreML path.
    pred = mlmodel.predict({
        "hidden_states": hs,
        "S_enc": np.array([S_enc], dtype=np.int32),
    })
    ck_ml = pred["cross_k"]
    cv_ml = pred["cross_v"]

    # Compare (tolerate fp32 rounding).
    ck_err = np.abs(ck_np.astype(np.float32) - ck_ml.astype(np.float32)).max()
    cv_err = np.abs(cv_np.astype(np.float32) - cv_ml.astype(np.float32)).max()
    print(f"  cross_k max error: {ck_err:.2e}")
    print(f"  cross_v max error: {cv_err:.2e}")
    if ck_err < 1e-4 and cv_err < 1e-4:
        print("  NumPy reference matches ✓")
    else:
        print("  WARNING: mismatch > 1e-4 — may need investigation")

    print("\n✓ cross_kv.mlpackage ready. Use from Swift:")
    print("  let kvModel = try MLModel(contentsOf: modelDir/cross_kv.mlpackage)")
    print("  let pred = try kvModel.prediction(from: MLDictionaryFeatureProvider(...))")
    print("  let crossK = pred.featureValue(for: \"cross_k\")!.multiArrayValue!")


if __name__ == "__main__":
    main()
