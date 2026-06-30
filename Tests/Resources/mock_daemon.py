import sys, json

ready = {"id": 0, "result": {"status": "ready", "model_loaded": True}}
sys.stdout.write(json.dumps(ready) + "\n")
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        resp = {"error": "invalid_json"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
        continue
    rid = req.get("id", 0)
    method = req.get("method", "")
    params = req.get("params", {})
    if method == "ping":
        resp = {"id": rid, "result": {"status": "ok", "model_loaded": True}}
    elif method == "cleanup":
        text = params.get("text", "")
        resp = {"id": rid, "result": {"cleaned": text + " [cleaned]"}}
    else:
        resp = {"id": rid, "error": "unknown_method: " + method}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
