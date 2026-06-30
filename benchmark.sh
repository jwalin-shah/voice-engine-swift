#!/bin/bash
# Download test audio samples and benchmark the voice pipeline.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="$DIR/test_data"
mkdir -p "$TEST_DIR"

info()  { echo -e "\033[0;32m[+]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[!]\033[0m $*"; }

# ── Download test samples ──────────────────────────────────────────────

if [ ! -f "$TEST_DIR/sample_2s.wav" ]; then
    info "Downloading LibriSpeech test-clean samples…"
    # Use a short utterance from OpenSLR (LibriSpeech test-clean).
    # These are 16kHz mono FLAC, we convert to WAV.
    BASE="https://www.openslr.org/resources/12/test-clean"
    for i in 1089 1342 1516 2128 2376 2606 2961 3223 3570 4507; do
        FLAC="$TEST_DIR/${i}.flac"
        WAV="$TEST_DIR/${i}.wav"
        if [ ! -f "$WAV" ]; then
            curl -sL "$BASE/$(printf '%d' $((i/100)))/$(printf '%d' $((i/10)))/${i}.flac" -o "$FLAC" 2>/dev/null || true
            if [ -f "$FLAC" ] && [ -s "$FLAC" ]; then
                ffmpeg -i "$FLAC" -ar 16000 -ac 1 -sample_fmt s16 "$WAV" -y -loglevel quiet 2>/dev/null
                rm "$FLAC"
                DUR=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$WAV")
                info "  $i → ${DUR}s"
            fi
        fi
    done
fi

# Also create synthetic test tones.
if [ ! -f "$TEST_DIR/silence_1s.wav" ]; then
    info "Creating synthetic test files…"
    python3 -c "
import wave, struct, math
# 1s silence
with wave.open('$TEST_DIR/silence_1s.wav', 'w') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(struct.pack('<' + 'h' * 16000, *([0]*16000)))
# 500Hz tone 2s
with wave.open('$TEST_DIR/tone_500hz_2s.wav', 'w') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    samples = [int(16000 * math.sin(2*math.pi*500*t/16000)) for t in range(32000)]
    w.writeframes(struct.pack('<' + 'h' * 32000, *samples))
print('  silence_1s.wav + tone_500hz_2s.wav')
" 2>&1
fi

# ── Count test files ────────────────────────────────────────────────────

info "Test files:"
for f in "$TEST_DIR"/*.wav; do
    DUR=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null || echo "?")
    SIZE=$(wc -c < "$f" | tr -d ' ')
    echo "  $(basename "$f")  ${DUR}s  ${SIZE} bytes"
done
