#!/usr/bin/env python3
"""
Export FullStop-large (XLM-RoBERTa token-classifier) to CoreML .mlpackage.

Requirements:
    pip install coremltools transformers torch sentencepiece

Pinned upstream revision:
    oliverguhr/fullstop-punctuation-multilang-large
    snapshot: 345e80adc07e761d3a35feafd20f2f44a151f453

Output:
    ~/.cache/fullstop-coreml/large/
        fullstop-punctuation.mlpackage     # CoreML model (fp16, mlprogram)
        tokenizer_compact.json             # Vocab with scores for Swift Viterbi tokenizer
        config.json                        # Labels, architecture, special tokens
        manifest.json                      # Provenance: revision, checksums, env
"""

import os, sys, json, hashlib, shutil
import numpy as np
import torch
import coremltools as ct
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
import sentencepiece as spm

# Pinned upstream revision
MODEL_ID = "oliverguhr/fullstop-punctuation-multilang-large"
SNAPSHOT_HASH = "345e80adc07e761d3a35feafd20f2f44a151f453"
MODEL_PATH = os.path.expanduser(
    f"~/.cache/huggingface/hub/models--oliverguhr--fullstop-punctuation-multilang-large/"
    f"snapshots/{SNAPSHOT_HASH}"
)
OUTPUT_DIR = Path.home() / ".cache" / "fullstop-coreml" / "large"

# Architecture constants (from config.json)
NUM_LABELS = 6
MAX_SEQ_LEN = 256
VOCAB_SIZE = 250002
ID2LABEL = {"0": "0", "1": ".", "2": ",", "3": "?", "4": "-", "5": ":"}

MANIFEST = {
    "model": MODEL_ID,
    "snapshot": SNAPSHOT_HASH,
    "model_size": "large",
    "num_labels": NUM_LABELS,
    "max_seq_len": MAX_SEQ_LEN,
    "converter": "coremltools",
    "compute_units": "CPU_AND_NE",
    "convert_to": "mlprogram",
    "deployment_target": "macOS15",
    "files": {},
}

# ── Export Tokenizer Data ──────────────────────────────────────────────────

def export_tokenizer_data(sp):
    """Export vocab with Unigram scores for Swift Viterbi tokenizer.

    Format (tokenizer_compact.json):
        {"vocab": [["<s>", 0.0], ["<pad>", 0.0], ...pieces ordered by HF ID...]}

    HF ID mapping for XLM-RoBERTa:
        HF 0 = <s>    (BOS)
        HF 1 = <pad>  (PAD)
        HF 2 = </s>   (EOS, from SPM ID 2)
        HF 3 = <unk>  (UNK, from SPM ID 0)
        HF 4+ = SPM piece 3..249999  (content pieces, SPM ID → HF ID = SPM ID + 1)
    """
    print("→ Exporting tokenizer data…")

    vocab_size = sp.GetPieceSize()  # 250000

    vocab = []
    # HF 0: BOS
    vocab.append(["<s>", 0.0])
    # HF 1: PAD
    vocab.append(["<pad>", 0.0])
    # HF 2: EOS (SPM ID 2)
    vocab.append(["</s>", float(sp.GetScore(2))])
    # HF 3: UNK (SPM ID 0)
    vocab.append(["<unk>", float(sp.GetScore(0))])
    # HF 4+: content pieces (SPM IDs 3..249999)
    for spm_id in range(3, vocab_size):
        piece = sp.IdToPiece(spm_id)
        score = float(sp.GetScore(spm_id))
        vocab.append([piece, score])

    # Write compact vocab
    compact_path = OUTPUT_DIR / "tokenizer_compact.json"
    with open(compact_path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab}, f, ensure_ascii=False)
    print(f"   tokenizer_compact.json: {len(vocab)} entries")

    # Also write id_to_piece.json for reference (same format Moonshine uses)
    id_to_piece = {}
    for hf_id, (piece, _score) in enumerate(vocab):
        id_to_piece[str(hf_id)] = piece
    id_path = OUTPUT_DIR / "id_to_piece.json"
    with open(id_path, "w", encoding="utf-8") as f:
        json.dump({"id_to_piece": id_to_piece}, f, ensure_ascii=False, indent=2)
    print(f"   id_to_piece.json: {len(id_to_piece)} entries")

    # Checksum for deterministic artifacts
    compact_data = json.dumps({"vocab": vocab}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    MANIFEST["files"]["tokenizer_compact.json"] = {
        "sha256": hashlib.sha256(compact_data).hexdigest(),
        "entries": len(vocab),
    }
    id_data = json.dumps({"id_to_piece": id_to_piece}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    MANIFEST["files"]["id_to_piece.json"] = {
        "sha256": hashlib.sha256(id_data).hexdigest(),
        "entries": len(id_to_piece),
    }

    # Copy SPM model for reference
    spm_path = os.path.join(MODEL_PATH, "sentencepiece.bpe.model")
    shutil.copy2(spm_path, OUTPUT_DIR / "sentencepiece.bpe.model")
    spm_hash = hashlib.sha256()
    with open(spm_path, "rb") as f:
        spm_hash.update(f.read())
    MANIFEST["files"]["sentencepiece.bpe.model"] = {
        "sha256": spm_hash.hexdigest(),
        "note": "Original SentencePiece Unigram model (reference, not loaded at runtime)",
    }


# ── CoreML Conversion ──────────────────────────────────────────────────────

class FullStopWrapper(torch.nn.Module):
    """Wrap XLMRobertaForTokenClassification to output only logits."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


def export_coreml():
    """Convert FullStop to CoreML .mlpackage with flexible sequence length."""
    print("→ Converting to CoreML…")

    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    wrapped = FullStopWrapper(model).eval()

    # Trace with example input
    batch_size = 1
    seq_len = MAX_SEQ_LEN
    dummy_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    dummy_input_ids[0, 0] = 0   # BOS
    dummy_input_ids[0, 1:] = 1  # PAD
    dummy_attention_mask[0, 0] = 1
    dummy_attention_mask[0, 1:] = 0

    traced = torch.jit.trace(
        wrapped,
        (dummy_input_ids, dummy_attention_mask),
        strict=False,
    )

    input_ids_type = ct.TensorType(
        name="input_ids",
        shape=(1, ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=MAX_SEQ_LEN)),
        dtype=np.int32,
    )
    attention_mask_type = ct.TensorType(
        name="attention_mask",
        shape=(1, ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=MAX_SEQ_LEN)),
        dtype=np.int32,
    )

    mlmodel = ct.convert(
        traced,
        inputs=[input_ids_type, attention_mask_type],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
    )

    # Save
    output_path = OUTPUT_DIR / "fullstop-punctuation.mlpackage"
    if output_path.exists():
        shutil.rmtree(output_path)
    mlmodel.save(str(output_path))
    print(f"   Model saved to {output_path}")

    # Hash stable payloads (weights inside mlpackage, not package metadata)
    weights_dir = output_path / "Data" / "com.apple.CoreML" / "weights"
    if weights_dir.exists():
        for wf in sorted(weights_dir.glob("*.bin")):
            h = hashlib.sha256()
            with open(wf, "rb") as f:
                h.update(f.read())
            MANIFEST["files"][f"model/weights/{wf.name}"] = {
                "sha256": h.hexdigest(),
                "size": wf.stat().st_size,
            }

    return mlmodel


# ── Verify (Python-side: CoreML vs PyTorch) ────────────────────────────────

def verify(mlmodel):
    """Compare CoreML output against PyTorch reference."""
    print("→ Verifying CoreML against PyTorch…")

    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

    test_texts = [
        "hello world how are you",
        "my name is clara and i live in berkeley california",
        "what time is it",
    ]

    for text in test_texts:
        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        encoded = tok(text, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN)

        # PyTorch reference
        with torch.no_grad():
            pt_logits = model(**encoded).logits.numpy()

        # CoreML inference
        coreml_input = {
            "input_ids": encoded["input_ids"].numpy().astype(np.int32),
            "attention_mask": encoded["attention_mask"].numpy().astype(np.int32),
        }
        coreml_out = mlmodel.predict(coreml_input)

        # mlprogram may return list or dict — normalize
        if isinstance(coreml_out, dict):
            cm_logits = coreml_out["logits"]
        elif isinstance(coreml_out, list):
            cm_logits = coreml_out[0] if isinstance(coreml_out[0], np.ndarray) else coreml_out
        else:
            cm_logits = np.array(coreml_out)

        pt_preds = np.argmax(pt_logits, axis=-1)
        cm_preds = np.argmax(cm_logits, axis=-1)
        agreement = (pt_preds == cm_preds).mean()
        max_diff = np.abs(pt_logits - cm_logits).max()

        status = "PASS" if agreement > 0.99 else "FAIL"
        print(f"   [{status}] '{text[:40]}...' — label agreement: {agreement:.4f}, max logit diff: {max_diff:.6f}")


# ── Config ─────────────────────────────────────────────────────────────────

def export_config():
    config = {
        "num_labels": NUM_LABELS,
        "id2label": ID2LABEL,
        "max_seq_len": MAX_SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "pad_token_id": 1,
        "unk_token_id": 3,
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    config_data = json.dumps(config, sort_keys=True).encode("utf-8")
    MANIFEST["files"]["config.json"] = {
        "sha256": hashlib.sha256(config_data).hexdigest(),
    }


# ── Manifest ───────────────────────────────────────────────────────────────

def write_manifest():
    import platform
    MANIFEST["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "coremltools": ct.__version__,
    }
    mpath = OUTPUT_DIR / "manifest.json"
    with open(mpath, "w") as f:
        json.dump(MANIFEST, f, indent=2)
    print(f"   Manifest: {mpath}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print(f"Download it first:")
        print(f"  python -c \"from transformers import AutoModel; AutoModel.from_pretrained('{MODEL_ID}')\"")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model: {MODEL_ID}")
    print(f"Snapshot: {SNAPSHOT_HASH}")
    print(f"Output directory: {OUTPUT_DIR}")

    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(MODEL_PATH, "sentencepiece.bpe.model"))

    export_tokenizer_data(sp)
    export_config()
    mlmodel = export_coreml()
    verify(mlmodel)
    write_manifest()

    print("\n✓ FullStop CoreML export complete")
    for name in sorted(MANIFEST["files"]):
        print(f"  {name}")


if __name__ == "__main__":
    main()
