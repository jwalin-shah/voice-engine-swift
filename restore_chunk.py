#!/usr/bin/env python3
"""Restore chunk_transcribe.py to 10s chunks, 2s overlap."""
path = "voice-engine-swift/chunk_transcribe.py"
with open(path) as f:
    c = f.read()
c = c.replace("CHUNK_SAMPLES = 5 * SR", "CHUNK_SAMPLES = 10 * SR")
c = c.replace("OVERLAP_SAMPLES = 1 * SR  # 1s overlap", "OVERLAP_SAMPLES = 2 * SR  # 2s overlap")
with open(path, "w") as f:
    f.write(c)
print("Restored: 10s chunks, 2s overlap")
