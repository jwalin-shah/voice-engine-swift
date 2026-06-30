#!/bin/bash
# Build and run the VoiceEngine Swift app.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
err()   { echo -e "${RED}[x]${NC} $*"; }

cmd_setup() {
    info "Exporting CoreML models (Python)…"
    command -v python3 >/dev/null || { err "python3 required"; exit 1; }

    pip install -q coremltools transformers moonshine-onnx torch sentencepiece numpy 2>/dev/null || true

    python3 "$DIR/Scripts/export_models.py" "$@"
    python3 "$DIR/Scripts/export_decoder.py" "$@"

    info "Models in ~/.cache/moonshine-coreml/tiny-streaming/"
    info "Run './build.sh build && ./build.sh run' to start."
}

cmd_build() {
    info "Building VoiceEngine (release)…"
    swift build -c release --package-path "$DIR"
    local bin="$DIR/.build/release/voice"
    if [ -f "$bin" ]; then
        info "Binary ready: $bin"
    fi
}

cmd_run() {
    # Kill any existing voice process before launching.
    pkill -f "$DIR/.build/release/voice" 2>/dev/null || true
    sleep 0.3

    if [ ! -f "$DIR/.build/release/voice" ]; then
        cmd_build
    fi
    info "Launching VoiceEngine menubar app…"
    "$DIR/.build/release/voice" &
    info "Press Caps Lock to dictate. Check menubar for mic icon."
}

cmd_bench() {
    info "Benchmark — transcribing 5s of silence…"
    cmd_build

    # Create a 5s silence WAV at 16kHz mono 16-bit.
    local wav="/tmp/bench_silence.wav"
    python3 -c "
import wave, struct
n = 5 * 16000
with wave.open('$wav', 'w') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(struct.pack('<' + 'h' * n, *([0]*n)))
print(f'{n} samples, 5.0s')
" 2>&1

    # Convert WAV to raw float32 and time the inference via the Swift binary.
    local raw="/tmp/bench_input.f32"
    python3 -c "
import wave, struct, sys
with wave.open('$wav', 'rb') as w:
    n = w.getnframes()
    samples = struct.unpack('<' + 'h' * n, w.readframes(n))
    f32 = [s / 32768.0 for s in samples]
    with open('$raw', 'wb') as f:
        for s in f32:
            f.write(struct.pack('f', s))
" 2>&1

    local start
    start=$(python3 -c 'import time; print(time.perf_counter())')
    "$DIR/.build/release/voice" --bench "$raw" 2>&1 || true
    local end
    end=$(python3 -c 'import time; print(time.perf_counter())')
    printf "[+] Wall clock: %.0f ms\n" "$(python3 -c "print(f'{(float($end)-float($start))*1000:.0f}')")"

    rm -f "$wav" "$raw"
}

cmd_clean() {
    info "Cleaning…"
    rm -rf "$DIR/.build"
}


cmd_test() {
    info "Running tests..."
    swift run -c debug voice-tests --package-path "$DIR" 2>&1
}

cmd_debug() {
    info "Building VoiceEngine (debug)..."
    swift build -c debug --package-path "$DIR" 2>&1
    local bin="$DIR/.build/debug/voice"
    if [ -f "$bin" ]; then
        info "Debug binary ready: $bin"
    fi
}

case "${1:-}" in
    setup)  shift; cmd_setup "$@";;
    build)  cmd_build;;
    debug)  cmd_debug;;
    test)   cmd_test;;
    run)    cmd_run;;
    clean)  cmd_clean;;
    *)
        echo "Usage: $0 {setup|build|debug|test|run|clean}"
        echo ""
        echo "  setup  Export CoreML models (Python + pip)"
        echo "  build  Compile Swift release binary"
        echo "  debug  Build debug binary"
        echo "  test   Run all tests"
        echo "  run    Build and launch menubar app"
        echo "  clean  Remove .build/"
        exit 1
        ;;
esac
