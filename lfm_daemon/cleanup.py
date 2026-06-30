import re
import mlx.core as mx
import mlx_lm

_model = None
_tokenizer = None
_loaded = False

CLEANUP_SYSTEM_PROMPT = (
    "You are a text cleanup assistant..."
    " Remove filler words (um, uh, like, you know),"
    " add proper punctuation and capitalization."
    " Keep all content. Never rephrase or summarize."
    " Output ONLY cleaned text.\n"
)

def load_model(model_path: str):
    global _model, _tokenizer, _loaded
    import os
    path = os.path.join(model_path, "mlx_model")
    _model, _tokenizer = mlx_lm.load(path)
    _ = mlx_lm.generate(_model, _tokenizer, "warm", max_tokens=5, verbose=False)
    _loaded = True
    return True

def is_loaded() -> bool:
    return _loaded

def cleanup_text(text: str, mode: str = "full") -> str:
    if not text or not text.strip():
        return text
    if not _loaded:
        return text
    messages = [
        {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
        {"role": "user", "content": f"Clean this: {text}"},
    ]
    prompt = _tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    response = mlx_lm.generate(
        _model, _tokenizer, prompt=prompt,
        max_tokens=max(len(text) * 3, 128),
        temperature=0.1, top_p=0.9,
        stop_strings=["<|im_end|>", "<|im_start|>"],
        verbose=False,
    )
    return response.strip().strip("\"")

FILLER_PATTERNS = [
    (r"\bum\b", ""), (r"\buh\b", ""),
    (r"\blike\b", ""), (r"\byou know\b", ""),
    (r"\bi mean\b", ""), (r"\bsort of\b", ""),
    (r"\bkind of\b", ""), (r"\bactually\b", ""),
    (r"\bbasically\b", ""), (r"\bliterally\b", ""),
    (r"\bright\b", ""), (r"\bso\b", ""),
]

def filler_only(text: str) -> str:
    result = text
    for p, r in FILLER_PATTERNS:
        result = re.sub(p, r, result, flags=re.IGNORECASE)
    result = re.sub(r"\s+", " ", result).strip()
    return result
