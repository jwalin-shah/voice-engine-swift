#!/usr/bin/env python3
import urllib.request, json
# Try different dataset names
names = ["librispeech", "librispeech_asr", "LibriSpeech", "hf-internal-testing/librispeech_asr_dummy"]
for name in names:
    url = f"https://datasets-server.huggingface.co/splits?dataset={name}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        print(f"{name}: OK")
        for s in data.get("splits", [])[:5]:
            print(f"  {s.get('config','?')} / {s.get('split','?')}: {s.get('num_examples',0)} rows")
        break
    except Exception as e:
        print(f"{name}: {e}")
