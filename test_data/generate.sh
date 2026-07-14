#!/bin/bash
# Generate test audio files for the voice-engine bench.
# Uses macOS built-in 'say' — no external dependencies.
set -euo pipefail

cd "$(dirname "$0")"

echo "Generating test audio..."

# AIFF (uncompressed) — preferred format for reproducible bench results
say -o quick_brown.aiff "the quick brown fox jumps over the lazy dog"
echo "  quick_brown.aiff — AIFF, ~2s speech"

# M4A (compressed) — second format to validate multi-format loading
say -o quick_brown.m4a --data-format aac "the quick brown fox jumps over the lazy dog"
echo "  quick_brown.m4a — AAC/M4A, ~2s speech"

echo "Done. Run: swift run voice-engine bench test_data/quick_brown.aiff"
