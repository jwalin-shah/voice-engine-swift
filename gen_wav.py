#!/usr/bin/env python3
"""Generate a mono 16kHz WAV file with tone + noise."""
import wave, struct, math, random, sys
dur_s = float(sys.argv[1])
path = sys.argv[2]
n = int(dur_s * 16000)
samples = []
for i in range(n):
    t = i / 16000.0
    v = 0.3 * math.sin(2 * math.pi * 440 * t) + 0.3 * math.sin(2 * math.pi * 880 * t)
    v += 0.05 * random.random()
    samples.append(max(-1, min(1, v)))
with wave.open(path, 'w') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    packed = b''.join(struct.pack('<h', int(s * 32767)) for s in samples)
    w.writeframes(packed)
print(f'{path}: {dur_s}s, {len(samples)} samples')
