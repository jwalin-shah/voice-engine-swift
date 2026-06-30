#!/usr/bin/env python3
"""Measure VoiceEngine inference with RAM stats.

Usage:
  ! python3 bench_full.py test.wav              # real audio
  ! python3 bench_full.py                       # synthetic 3s
  ! python3 bench_full.py --duration 15          # synthetic 15s
  ! python3 bench_full.py --iterations 5         # average
  ! python3 bench_full.py --json                 # machine-readable
"""
import subprocess, sys, os, json, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

args = " ".join(sys.argv[1:])
cmd = f"python3 bench.py {args}"
start = time.perf_counter()
proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
elapsed = time.perf_counter() - start

# Parse JSON output
output = proc.stdout
result = {}
try:
    # Find JSON block in output
    lines = output.strip().split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("{"):
            result = json.loads("\n".join(lines[i:]))
            break
except: pass

# RAM measurement
try:
    import psutil
    mem = psutil.Process(os.getpid()).memory_info()
    result["ram_rss_mb"] = round(mem.rss / 1024 / 1024, 1)
    result["ram_vms_mb"] = round(mem.vms / 1024 / 1024, 1)
except ImportError:
    vm = subprocess.run(["ps", "-o", "rss=", str(os.getpid())], capture_output=True, text=True)
    rss_kb = vm.stdout.strip()
    result["ram_rss_mb"] = round(int(rss_kb) / 1024, 1) if rss_kb.isdigit() else "N/A"

result["wall_clock_ms"] = round(elapsed * 1000, 1)
print(json.dumps(result, indent=2))
