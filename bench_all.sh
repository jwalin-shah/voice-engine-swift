#!/bin/bash
# VoiceEngine full benchmark suite
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
MC="$HOME/.cache/moonshine-coreml/tiny-streaming"
BACKUP="$HOME/.cache/moonshine-coreml/tiny-streaming-cpu-only"
WAV3="/tmp/voice_3s.wav"
WAV5="/tmp/voice_5s.wav"
WAV10="/tmp/voice_10s.wav"
CPU_RESULT="/tmp/voice_cpu_result.json"
GPU_RESULT="/tmp/voice_gpu_result.json"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info(){ echo -e "${GREEN}[+]${NC} $*"; }
header(){ echo -e "\n${CYAN}━━━ $* ━━━${NC}"; }

header "1. Generate test audio"
python3 "$DIR/gen_wav.py" 3 "$WAV3"
python3 "$DIR/gen_wav.py" 5 "$WAV5"
python3 "$DIR/gen_wav.py" 10 "$WAV10"

header "2. CPU_ONLY decoder baseline"
python3 "$DIR/bench.py" "$WAV3" --iterations 5 --json | tee "$CPU_RESULT"

header "3. Backup CPU_ONLY decoder and export GPU"
if [ -d "$MC/decoder_stateful.mlpackage" ]; then
    mkdir -p "$BACKUP"
    rm -rf "$BACKUP/decoder_stateful.mlpackage" 2>/dev/null || true
    cp -R "$MC/decoder_stateful.mlpackage" "$BACKUP/decoder_stateful.mlpackage"
    info "Backed up to $BACKUP"
fi

header "4. Export decoder (CPU_AND_GPU)"
python3 "$DIR/Scripts/export_decoder_gpu.py" 2>&1

header "5. Benchmark GPU decoder"
python3 "$DIR/bench.py" "$WAV3" --iterations 5 --json | tee "$GPU_RESULT"

header "6. Longer audio on GPU"
python3 "$DIR/bench.py" "$WAV5" --iterations 3 --json
python3 "$DIR/bench.py" "$WAV10" --iterations 3 --json

header "7. RAM measurement"
# Start bench process, sample RSS
python3 "$DIR/bench.py" "$WAV10" --iterations 1 --json &
BENCH_PID=$!
sleep 1.5
RSS_KB=$(ps -o rss= -p "$BENCH_PID" 2>/dev/null || echo 0)
wait "$BENCH_PID" 2>/dev/null || true
RSS_MB=$(echo "scale=1; $RSS_KB / 1024" | bc)
echo "  Peak RSS: ${RSS_MB} MB"

header "8. Restore CPU_ONLY"
rm -rf "$MC/decoder_stateful.mlpackage"
cp -R "$BACKUP/decoder_stateful.mlpackage" "$MC/decoder_stateful.mlpackage"
info "Restored CPU_ONLY"

header "9. Comparison"
CPU_LOOP=$(python3 -c "import json; d=json.load(open('$CPU_RESULT')); print(d.get('decoder_loop',{}).get('mean_ms',0))")
GPU_LOOP=$(python3 -c "import json; d=json.load(open('$GPU_RESULT')); print(d.get('decoder_loop',{}).get('mean_ms',0))")
CPU_TOTAL=$(python3 -c "import json; d=json.load(open('$CPU_RESULT')); t=sum(d.get(k,{}).get('mean_ms',0) for k in ['preprocess','encoder','kv_projection','decoder_loop','token_decode']); print(t)")
GPU_TOTAL=$(python3 -c "import json; d=json.load(open('$GPU_RESULT')); t=sum(d.get(k,{}).get('mean_ms',0) for k in ['preprocess','encoder','kv_projection','decoder_loop','token_decode']); print(t)")
echo "  CPU decoder: ${CPU_LOOP}ms loop, ${CPU_TOTAL}ms total"
echo "  GPU decoder: ${GPU_LOOP}ms loop, ${GPU_TOTAL}ms total"
if python3 -c "exit(0 if float('$GPU_LOOP') > 0 else 1)" 2>/dev/null; then
    RATIO=$(python3 -c "print(f'{float($CPU_LOOP)/float($GPU_LOOP):.1f}x')")
    echo "  Speedup: $RATIO (decoder loop)"
fi
echo "  RAM: ${RSS_MB:-N/A} MB peak"
echo -e "\n${GREEN}✓ Done${NC}"
