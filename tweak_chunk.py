#!/usr/bin/env python3
"""Replace CHUNK_SAMPLES and OVERLAP_SAMPLES in chunk_transcribe.py."""
path = "voice-engine-swift/chunk_transcribe.py"
with open(path) as f:
    c = f.read()
c = c.replace("CHUNK_SAMPLES = 10 * SR", "CHUNK_SAMPLES = 5 * SR")
c = c.replace("OVERLAP_SAMPLES = 2 * SR  # 2s overlap", "OVERLAP_SAMPLES = 1 * SR  # 1s overlap")
with open(path, "w") as f:
    f.write(c)
print("Updated: 5s chunks, 1s overlap")
