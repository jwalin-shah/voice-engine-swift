import sys, json, signal
from pathlib import Path
import cleanup

VERSION = "1.0.0"
_model_path = None
_cleanup_available = False

def handle_ping(params):
    return {
        "status": "ok",
        "version": VERSION,
        "model_loaded": cleanup.is_loaded(),
        "cleanup_available": _cleanup_available,
    }

def handle_cleanup(params):
    text = params.get("text", "")
    mode = params.get("mode", "full")
    if mode == "disabled":
        return {"cleaned": text}
    if mode == "filler_only":
        cleaned = cleanup.filler_only(text)
        return {"cleaned": cleaned}
    cleaned = cleanup.cleanup_text(text, mode)
    return {"cleaned": cleaned}

HANDLERS = {
    "ping": handle_ping,
    "cleanup": handle_cleanup,
}

def process_request(raw: str) -> str:
    try:
        req = json.loads(raw)
    except json.JSONDecodeError:
        return json.dumps({"error": "invalid_json"})
    rid = req.get("id", 0)
    method = req.get("method", "")
    params = req.get("params", {})
    handler = HANDLERS.get(method)
    if handler is None:
        return json.dumps({"id": rid, "error": f"unknown_method: {method}"})
    try:
        result = handler(params)
        return json.dumps({"id": rid, "result": result})
    except Exception as e:
        return json.dumps({"id": rid, "error": str(e)})

def main():
    global _model_path, _cleanup_available
    _model_path = Path(__file__).parent.parent.resolve()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    try:
        cleanup.load_model(str(_model_path))
        _cleanup_available = True
        sys.stderr.write("[daemon] Model loaded successfully\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[daemon] Model load failed: {e}\n")
        sys.stderr.flush()
        _cleanup_available = False
    sys.stdout.write(json.dumps({"id": 0, "result": {"status": "ready", "model_loaded": _cleanup_available}}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = process_request(line)
            sys.stdout.write(response + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"[daemon] Fatal: {e}\n")
            sys.stderr.flush()

if __name__ == "__main__":
    main()
