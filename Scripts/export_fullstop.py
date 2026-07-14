#!/usr/bin/env python3
"""
Export FullStop-large (XLM-RoBERTa token-classifier) to CoreML .mlpackage.

Requirements:
    pip install coremltools transformers torch sentencepiece

Output:
    ~/.cache/fullstop-coreml/large/
        fullstop-punctuation.mlpackage     # CoreML model (fp16)
        id_to_piece.json                    # Vocab: id → piece string
        merges.json                         # BPE merge table for Swift tokenizer
        config.json                         # Labels, architecture, special tokens
        sentencepiece.bpe.model             # Original SPM model (reference)
"""

import os, sys, json, shutil
import numpy as np
import torch
import coremltools as ct
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
import sentencepiece as spm

MODEL_ID = "oliverguhr/fullstop-punctuation-multilang-large"
# Use locally cached model
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--oliverguhr--fullstop-punctuation-multilang-large/"
    "snapshots/345e80adc07e761d3a35feafd20f2f44a151f453"
)
OUTPUT_DIR = Path.home() / ".cache" / "fullstop-coreml" / "large"

# Architecture constants (from config.json)
NUM_LABELS = 6
MAX_SEQ_LEN = 256  # flexible, use this as default trace length
VOCAB_SIZE = 250002
ID2LABEL = {"0": "0", "1": ".", "2": ",", "3": "?", "4": "-", "5": ":"}

# ── Export Tokenizer Data ──────────────────────────────────────────────────────

def export_tokenizer_data():
    """Extract vocabulary and BPE merge rules from SPM model for Swift tokenizer."""
    print("→ Exporting tokenizer data…")

    spm_path = os.path.join(MODEL_PATH, "sentencepiece.bpe.model")
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)

    # 1. id_to_piece.json — same format Moonshine uses, with HF ID mapping
    # HF tokenizer shifts SPM IDs by +1 for content tokens (SPM[0]=UNK→HF[3], SPM[1]=BOS→unused, etc.)
    # For the Swift tokenizer, we build a clean HF-compatible mapping.
    id_to_piece = {}
    vocab_size = sp.GetPieceSize()  # 250000

    for spm_id in range(vocab_size):
        piece = sp.IdToPiece(spm_id)
        # Map SPM ID → HF ID:
        # SPM  0 (UNK)   → HF  3
        # SPM  1 (BOS)   → skip (HF uses 0 for BOS, separate)
        # SPM  2 (EOS)   → HF  2
        # SPM  3..249999 → HF  4..250000
        if spm_id == 0:  # UNK
            hf_id = 3
        elif spm_id == 1:  # BOS — skip, HF uses id=0
            continue
        elif spm_id == 2:  # EOS
            hf_id = 2
        else:
            hf_id = spm_id + 1

        id_to_piece[str(hf_id)] = piece

    # Add special tokens
    id_to_piece["0"] = "<s>"     # HF BOS
    id_to_piece["1"] = "<pad>"   # HF PAD
    # 2 is EOS already covered
    # 3 is UNK already covered

    with open(OUTPUT_DIR / "id_to_piece.json", "w", encoding="utf-8") as f:
        json.dump({"id_to_piece": id_to_piece}, f, ensure_ascii=False, indent=2)

    # 2. BPE merge table for Swift encoder
    # SentencePiece BPE uses a merge table: (left_piece, right_piece) → new_piece_id
    # We export the piece strings and merge priorities for Swift-side encoding.
    # Format: list of [left_piece, right_piece, merged_piece]
    # These are ordered by merge priority (first = highest).
    merges = []
    for spm_id in range(vocab_size):
        piece = sp.IdToPiece(spm_id)
        score = sp.GetScore(spm_id)
        # Only include pieces with valid scores (actual BPE merges, not base characters)
        if score > 0 and piece not in ('<unk>', '<s>', '</s>', '<pad>'):
            # The merge pair is encoded in the piece itself
            # For BPE, each merge creates a new piece from two subpieces
            merges.append({"piece": piece, "score": float(score), "spm_id": spm_id})

    with open(OUTPUT_DIR / "merges.json", "w", encoding="utf-8") as f:
        json.dump(merges, f, ensure_ascii=False, indent=2)

    # 3. Copy the original SPM model for reference
    shutil.copy2(spm_path, OUTPUT_DIR / "sentencepiece.bpe.model")

    print(f"   Vocab size: {vocab_size} SPM pieces → {len(id_to_piece)} HF entries")
    print(f"   Merges: {len(merges)}")

    return sp


# ── CoreML Conversion ──────────────────────────────────────────────────────────

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

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
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
    # Set first position to BOS, fill rest with PAD
    dummy_input_ids[0, 0] = 0  # BOS
    dummy_input_ids[0, 1:] = 1  # PAD
    dummy_attention_mask[0, 0] = 1
    dummy_attention_mask[0, 1:] = 0

    traced = torch.jit.trace(
        wrapped,
        (dummy_input_ids, dummy_attention_mask),
        strict=False,
    )

    # Define input types for CoreML
    # Use flexible sequence length: 1..256
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

    # Also save config for Swift-side reference
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

    print(f"   Model saved to {output_path}")
    return mlmodel


# ── Verify ──────────────────────────────────────────────────────────────────────

def verify(mlmodel, sp):
    """Compare CoreML output against PyTorch reference."""
    print("→ Verifying CoreML against PyTorch…")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

    test_texts = [
        "hello world how are you",
        "my name is clara and i live in berkeley california",
        "what time is it",
    ]

    for text in test_texts:
        encoded = tok(text, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = encoded['input_ids']
        attn_mask = encoded['attention_mask']
        seq_len = input_ids.shape[1]

        # PyTorch reference
        with torch.no_grad():
            pt_logits = model(**encoded).logits.numpy()

        # CoreML inference
        coreml_input = {
            "input_ids": input_ids.numpy().astype(np.int32),
            "attention_mask": attn_mask.numpy().astype(np.int32),
        }
        coreml_out = mlmodel.predict(coreml_input)
        # mlprogram may return list or dict — normalize
        if isinstance(coreml_out, dict):
            cm_logits = coreml_out["logits"]
        elif isinstance(coreml_out, list):
            cm_logits = coreml_out[0] if isinstance(coreml_out[0], np.ndarray) else coreml_out
        else:
            cm_logits = np.array(coreml_out)

        # Compare: check argmax agreement (labels match)
        pt_preds = np.argmax(pt_logits, axis=-1)
        cm_preds = np.argmax(cm_logits, axis=-1)
        agreement = (pt_preds == cm_preds).mean()

        # Also check max logit difference
        max_diff = np.abs(pt_logits - cm_logits).max()

        status = "PASS" if agreement > 0.99 else "FAIL"
        print(f"   [{status}] '{text[:40]}...' — label agreement: {agreement:.4f}, max logit diff: {max_diff:.6f}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    sp = export_tokenizer_data()
    mlmodel = export_coreml()
    verify(mlmodel, sp)

    print("\n✓ FullStop CoreML export complete")
    print(f"  Model: {OUTPUT_DIR / 'fullstop-punctuation.mlpackage'}")
    print(f"  Vocab: {OUTPUT_DIR / 'id_to_piece.json'}")
    print(f"  Merges: {OUTPUT_DIR / 'merges.json'}")
    print(f"  Config: {OUTPUT_DIR / 'config.json'}")


if __name__ == "__main__":
    main()
