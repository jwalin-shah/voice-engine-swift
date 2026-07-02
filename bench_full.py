#!/usr/bin/env python3
"""Measure VoiceEngine inference with RAM stats.

Usage:
  ! python3 bench_full.py test.wav              # real audio
  ! python3 bench_full.py                       # synthetic 3s
  ! python3 bench_full.py --duration 15          # synthetic 15s
  ! python3 bench_full.py --iterations 5         # average
  ! python3 bench_full.py --json                 # machine-readable
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BENCH = REPO_ROOT / "bench.py"


def parse_json_block(output):
    """Return the first trailing JSON object emitted by bench.py."""
    lines = output.strip().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("{"):
            try:
                return json.loads("\n".join(lines[i:]))
            except json.JSONDecodeError:
                continue
    return {}


def rss_mb(pid):
    """Sample resident set size for pid in MiB, or None if unavailable."""
    proc = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
    )
    rss_kb = proc.stdout.strip()
    return round(int(rss_kb) / 1024, 1) if rss_kb.isdigit() else None


def main():
    bench_args = sys.argv[1:]
    if "--json" not in bench_args:
        bench_args.append("--json")

    cmd = [sys.executable, str(BENCH), *bench_args]
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    peak_rss = 0.0
    while proc.poll() is None:
        sample = rss_mb(proc.pid)
        if sample is not None:
            peak_rss = max(peak_rss, sample)
        time.sleep(0.05)

    stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - start
    final_rss = rss_mb(proc.pid)
    if final_rss is not None:
        peak_rss = max(peak_rss, final_rss)

    result = parse_json_block(stdout)
    result["ram_peak_rss_mb"] = round(peak_rss, 1) if peak_rss else "N/A"
    result["wall_clock_ms"] = round(elapsed * 1000, 1)
    result["returncode"] = proc.returncode
    if proc.returncode != 0:
        result["stdout_tail"] = stdout.strip()[-1000:]
        result["stderr_tail"] = stderr.strip()[-1000:]

    print(json.dumps(result, indent=2))
    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
