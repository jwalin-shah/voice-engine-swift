#!/usr/bin/env python3
"""Export parakeet-tdt-0.6b-v2-mlx weights from safetensors to flat .f32 files.

Usage:
    python3 export_weights.py [--output-dir DIR] [--model-id MODEL_ID]

Default model: senstella/parakeet-tdt-0.6b-v2-mlx
Default output: Sources/VoiceEngine/ParakeetMLX/weights/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open


def export_weights(model_path: str, output_dir: str):
    """Export all weights to .f32 files and write weight_manifest.json."""
    os.makedirs(output_dir, exist_ok=True)

    manifest = {}
    total_size = 0

    with safe_open(model_path, framework="np") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # MLX weights are already in the right layout (no transpose needed).
            # But we need to handle some shapes for C++ loading:
            # - Conv1d weights in MLX: (out_channels, kernel_size, in_channels) -> need to track shape
            # - Conv2d weights in MLX: (out_channels, kh, kw, in_channels) -> need to track shape
            # - Linear weights in MLX: (out_features, in_features) -> need to track shape
            # Use float16 to match MLX GPU inference (bfloat16 source weights)
            arr = tensor.astype(np.float32)  # intermediate: bfloat16 → fp32 for numpy
            arr = arr.astype(np.float16)     # export as float16

            # Use safe filename: replace dots with underscores
            fname = key.replace(".", "_") + ".f16"
            fpath = os.path.join(output_dir, fname)

            arr.tofile(fpath)
            size_bytes = arr.nbytes
            total_size += size_bytes

            manifest[key] = {
                "file": fname,
                "shape": list(arr.shape),
                "dtype": "float16",
                "size_bytes": size_bytes,
            }

    manifest_path = os.path.join(output_dir, "weight_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "num_tensors": len(manifest),
                "weights": manifest,
            },
            f,
            indent=2,
        )

    print(f"Exported {len(manifest)} tensors ({total_size / (1024**3):.2f} GB) to {output_dir}")
    print(f"Manifest: {manifest_path}")

    return window


def export_vocab(config_path: str, output_dir: str):
    """Extract vocabulary from config.json and save as vocab.json."""
    with open(config_path) as f:
        config = json.load(f)

    # The vocabulary is in the config under 'labels' array
    labels = config.get("labels", [])
    if not labels:
        print("WARNING: No 'labels' found in config.json")
        # Try training config format (NeMo)
        # The vocabulary is embedded in the decoder/joint config
        joint = config.get("joint", {})
        vocabulary = joint.get("vocabulary", [])
        if vocabulary:
            labels = vocabulary

    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"vocabulary": labels, "vocab_size": len(labels)}, f, indent=2)

    print(f"Vocabulary: {len(labels)} tokens -> {vocab_path}")


def main():
    parser = argparse.ArgumentParser(description="Export parakeet weights to .f32")
    parser.add_argument(
        "--model-id",
        default="senstella/parakeet-tdt-0.6b-v2-mlx",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for .f32 files",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.expanduser("~/.cache/huggingface/hub"),
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Direct path to model.safetensors (overrides model-id lookup)",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Direct path to config.json (for vocab extraction)",
    )
    args = parser.parse_args()

    # Resolve model path
    if args.model_path:
        safetensors_path = args.model_path
    else:
        # Resolve from HF cache
        model_dir = os.path.join(
            args.cache_dir,
            "models--" + args.model_id.replace("/", "--"),
        )
        snapshots_dir = os.path.join(model_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            print(f"ERROR: Model not found in cache: {model_dir}")
            print("Download it first: python3 -c \"from parakeet_mlx import from_pretrained; from_pretrained('{args.model_id}')\"")
            sys.exit(1)

        # Find the snapshot directory (there should be exactly one)
        snapshots = sorted(os.listdir(snapshots_dir))
        if not snapshots:
            print(f"ERROR: No snapshots found in {snapshots_dir}")
            sys.exit(1)

        snapshot_dir = os.path.join(snapshots_dir, snapshots[0])
        safetensors_path = os.path.join(snapshot_dir, "model.safetensors")
        if not os.path.isfile(safetensors_path):
            print(f"ERROR: model.safetensors not found at {safetensors_path}")
            sys.exit(1)

    # Resolve config path
    if args.config_path:
        config_path = args.config_path
    elif args.model_path:
        config_path = os.path.join(os.path.dirname(args.model_path), "config.json")
    else:
        config_path = os.path.join(snapshot_dir, "config.json")

    # Resolve output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default: project-relative
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir)) if "Scripts" in script_dir else os.getcwd()
        output_dir = os.path.join(repo_root, "Sources", "VoiceEngine", "ParakeetMLX", "weights")

    print(f"Model: {safetensors_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")

    export_weights(safetensors_path, output_dir)

    if os.path.isfile(config_path):
        export_vocab(config_path, output_dir)

    # Export model architecture config
    export_model_config(config_path, output_dir)

    # Export mel filterbank
    export_filterbank(config_path, output_dir)
    export_hanning_window(config_path, output_dir)

    # Also export the filterbank (mel basis) - needed for audio preprocessing
    print("\nMel filterbank note:")
    print("  The mel filterbank is computed from preprocessor config at runtime.")
    print("  Preprocessor params (from config.json preprocessor section):")
    with open(config_path) as f:
        config = json.load(f)
    preproc = config.get("preprocessor", {})
    print(f"    sample_rate: {preproc.get('sample_rate', 16000)}")
    print(f"    window_size: {preproc.get('window_size', 0.025)}")
    print(f"    window_stride: {preproc.get('window_stride', 0.01)}")
    print(f"    window: {preproc.get('window', 'hann')}")
    print(f"    features: {preproc.get('features', 80)}")
    print(f"    n_fft: {preproc.get('n_fft', 512)}")
    print(f"    normalize: {preproc.get('normalize', 'per_feature')}")
    print(f"    dither: {preproc.get('dither', 1e-05)}")


def export_model_config(config_path: str, output_dir: str):
    """Extract and save a simplified model architecture config for C++."""
    with open(config_path) as f:
        config = json.load(f)

    preproc = config.get("preprocessor", {})
    encoder = config.get("encoder", {})
    decoder_cfg = config.get("decoder", {})
    joint_cfg = config.get("joint", {})
    decoding = config.get("decoding", {})

    model_config = {
        "model_type": "tdt",
        # Preprocessor
        "sample_rate": preproc.get("sample_rate", 16000),
        "n_mels": preproc.get("features", 128),
        "n_fft": preproc.get("n_fft", 512),
        "hop_length": int(preproc.get("window_stride", 0.01) * preproc.get("sample_rate", 16000)),
        "win_length": int(preproc.get("window_size", 0.025) * preproc.get("sample_rate", 16000)),
        "window_fn": preproc.get("window", "hann"),
        "normalize": preproc.get("normalize", "per_feature"),
        "preemph": 0.97,  # default in parakeet_mlx
        # Encoder
        "enc_n_layers": encoder.get("n_layers", 24),
        "enc_d_model": encoder.get("d_model", 1024),
        "enc_n_heads": encoder.get("n_heads", 8),
        "enc_ff_expansion_factor": encoder.get("ff_expansion_factor", 4),
        "enc_conv_kernel_size": encoder.get("conv_kernel_size", 9),
        "enc_subsampling_factor": encoder.get("subsampling_factor", 8),
        "enc_subsampling_conv_channels": encoder.get("subsampling_conv_channels", 256),
        "enc_self_attention_model": encoder.get("self_attention_model", "rel_pos"),
        "enc_use_bias": encoder.get("use_bias", False),
        "enc_pos_emb_max_len": encoder.get("pos_emb_max_len", 5000),
        # Decoder (predict network)
        "pred_hidden": decoder_cfg.get("prednet", {}).get("pred_hidden", 640),
        "pred_rnn_layers": decoder_cfg.get("prednet", {}).get("pred_rnn_layers", 2),
        "vocab_size": decoder_cfg.get("vocab_size", 1024),
        "blank_as_pad": decoder_cfg.get("blank_as_pad", True),
        # Joint network
        "joint_hidden": joint_cfg.get("jointnet", {}).get("joint_hidden", 640),
        "joint_activation": joint_cfg.get("jointnet", {}).get("activation", "relu"),
        "num_extra_outputs": joint_cfg.get("num_extra_outputs", 5),
        "num_classes": joint_cfg.get("num_classes", 1024),
        # Decoding
        "durations": decoding.get("durations", [0, 1, 2, 3, 4]),
        "max_symbols": decoding.get("greedy", {}).get("max_symbols", 10),
    }

    model_config_path = os.path.join(output_dir, "model_config.json")
    with open(model_config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Model config: {model_config_path}")


def export_filterbank(config_path: str, output_dir: str):
    """Precompute mel filterbank and save as .f32."""
    with open(config_path) as f:
        config = json.load(f)

    preproc = config.get("preprocessor", {})
    sample_rate = preproc.get("sample_rate", 16000)
    n_fft = preproc.get("n_fft", 512)
    n_mels = preproc.get("features", 128)

    try:
        import librosa
    except ImportError:
        print("librosa not available, skipping filterbank export")
        print("Install with: pip install librosa")
        return

    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0,
        fmax=sample_rate / 2,
        norm="slaney",
    )
    # mel_basis shape: (n_mels, n_fft//2 + 1)
    mel_basis = mel_basis.astype(np.float32)

    output_path = os.path.join(output_dir, "mel_filterbank.f32")
    mel_basis.tofile(output_path)
    print(f"Mel filterbank: {mel_basis.shape} -> {output_path}")

    # Also save shape info
    shape_path = os.path.join(output_dir, "mel_filterbank_shape.txt")
    with open(shape_path, "w") as f:
        f.write(f"{mel_basis.shape[0]} {mel_basis.shape[1]}\n")


def export_hanning_window(config_path: str, output_dir: str):
    """Precompute Hanning window and save as .f32."""
    with open(config_path) as f:
        config = json.load(f)

    preproc = config.get("preprocessor", {})
    sample_rate = preproc.get("sample_rate", 16000)
    window_size = preproc.get("window_size", 0.025)
    win_length = int(window_size * sample_rate)

    window = np.hanning(win_length + 1)[:-1].astype(np.float32)

    output_path = os.path.join(output_dir, "hanning_window.f32")
    window.tofile(output_path)
    print(f"Hanning window: {window.shape} -> {output_path}")

    # Also generate C++ weight entries header
    _generate_weight_entries(manifest_from_weights, output_dir)


def _generate_weight_entries(manifest: dict, output_dir: str):
    for key, info in sorted(manifest.items()):
        fname = info["file"]
        shape_str = "{" + ", ".join(str(d) for d in info["shape"]) + "}"
        # Escape backslashes and quotes for C++ string
        key_escaped = key.replace("\\", "\\\\").replace('"', '\\"')
        fname_escaped = fname.replace("\\", "\\\\").replace('"', '\\"')
        entries.append(f'    {{"{key_escaped}", "{fname_escaped}", {shape_str}}}')

    cpp_source = f"""// Auto-generated by export_parakeet_weights.py — DO NOT EDIT.
// Weight entries for parakeet model.

#pragma once
#include <string>
#include <vector>

namespace parakeet {{

struct WeightEntry {{
    const char* key;
    const char* filename;
    std::vector<int> shape;
}};

static const std::vector<WeightEntry> kWeightEntries = {{
{','.join(entries)}
}};

static const int kNumWeights = {len(entries)};

}} // namespace parakeet
"""
    output_path = os.path.join(output_dir, "weight_entries.h")
    with open(output_path, "w") as f:
        f.write(cpp_source)
    print(f"C++ weight entries: {output_path}")


if __name__ == "__main__":
    main()
