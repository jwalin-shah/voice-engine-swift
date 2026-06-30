#!/bin/bash
# VoiceEngine diagnostic — checks if everything is ready to run
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; }
warn() { echo -e "${YELLOW}~${NC} $1"; }
header() { echo ""; echo "── $1 ──"; }

header "1. CoreML Models"
MODEL_DIR="$HOME/.cache/moonshine-coreml/tiny-streaming"
if [ -d "$MODEL_DIR" ]; then
    pass "Model directory exists: $MODEL_DIR"
    for f in encoder.mlpackage decoder_stateful.mlpackage config.json id_to_piece.json; do
        if [ -e "$MODEL_DIR/$f" ]; then
            pass "  $f found"
        else
            fail "  $f MISSING"
        fi
    done
else
    fail "Model directory NOT FOUND at $MODEL_DIR"
    warn "Run: cd voice-engine-swift && ./build.sh setup"
fi

header "2. Swift Binary"
BINARY="$DIR/.build/release/voice"
if [ -f "$BINARY" ]; then
    SIZE=$(stat -f%z "$BINARY" 2>/dev/null || echo "?")
    pass "Binary exists ($(echo "scale=1; $SIZE/1048576" | bc 2>/dev/null)MB)"
else
    fail "Binary NOT FOUND at $BINARY"
    warn "Run: cd voice-engine-swift && ./build.sh build"
fi

header "3. Python Dependencies"
for pkg in coremltools torch transformers moonshine-onnx sentencepiece numpy; do
    if python3 -c "import $pkg" 2>/dev/null; then
        pass "python3: $pkg"
    else
        fail "python3: $pkg NOT INSTALLED"
    fi
done

header "4. Permissions"
# Check if accessibility is enabled (crude: can we create a CGEvent tap?)
PERMS=$(python3 -c "
import sys
try:
    import Quartz
    # Try creating an event tap (will fail if accessibility denied)
    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionDefault,
        Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged),
        lambda *a: None,
        None
    )
    if tap is not None:
        print('accessibility_granted')
        Quartz.CFMachPortInvalidate(tap)
    else:
        print('accessibility_denied')
except Exception as e:
    print(f'error: {e}')
" 2>&1)
if [ "$PERMS" = "accessibility_granted" ]; then
    pass "Accessibility permission: GRANTED"
else
    warn "Accessibility: check System Settings → Privacy → Accessibility"
    echo "  (needed for Caps Lock hotkey and text paste)"
fi

header "5. macOS Version"
SW_VERS=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
MAJOR=$(echo "$SW_VERS" | cut -d. -f1)
if [ "$MAJOR" -ge 14 ]; then
    pass "macOS $SW_VERS"
else
    warn "macOS $SW_VERS — needs 14+ for ANE/CoreML support"
fi

header "Summary"
if [ -f "$BINARY" ] && [ -d "$MODEL_DIR" ]; then
    echo -e "${GREEN}Ready to run!${NC}"
    echo "  cd voice-engine-swift && ./build.sh run"
elif [ -f "$BINARY" ] && [ ! -d "$MODEL_DIR" ]; then
    echo -e "${YELLOW}Need models. Setup:${NC}"
    echo "  cd voice-engine-swift && ./build.sh setup && ./build.sh run"
elif [ ! -f "$BINARY" ] && [ -d "$MODEL_DIR" ]; then
    echo -e "${YELLOW}Need build:${NC}"
    echo "  cd voice-engine-swift && ./build.sh build && ./build.sh run"
else
    echo -e "${YELLOW}Need both setup and build:${NC}"
    echo "  cd voice-engine-swift && ./build.sh setup && ./build.sh run"
fi
