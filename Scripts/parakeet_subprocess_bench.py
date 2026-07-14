#!/usr/bin/env python3
"""
Benchmark: subprocess worker communication overhead.

Spawns parakeet_worker.py as a persistent subprocess, sends audio paths,
measures end-to-end latency (Swift perspective) vs worker-internal latency.

The difference is IPC (JSON serialization + pipe I/O + deserialization) overhead.
"""

import json
import subprocess
import sys
import time
import glob
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    print("=" * 60, file=sys.stderr)
    print("Parakeet MLX Subprocess Overhead Benchmark", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    wav_dir = PROJECT_ROOT / "bench_data" / "external_audio"
    wav_paths = sorted(glob.glob(str(wav_dir / "*.wav")))[:20]

    if not wav_paths:
        print("ERROR: No WAV files", file=sys.stderr)
        sys.exit(1)

    worker_script = str(PROJECT_ROOT / "Scripts" / "parakeet_worker.py")
    venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python3")

    print(f"\nLaunching worker: {venv_python} {worker_script}", file=sys.stderr)

    proc = subprocess.Popen(
        [venv_python, "-u", worker_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    ready = threading.Event()

    def read_stderr():
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                print(f"  [worker] {line}", file=sys.stderr)
            if "ready" in line:
                ready.set()

    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stderr_thread.start()

    if not ready.wait(timeout=60):
        print("ERROR: worker did not become ready", file=sys.stderr)
        proc.kill()
        sys.exit(1)

    print("\nWorker ready. Running warmup + benchmark...\n", file=sys.stderr)

    # Warmup: 3 calls
    for i in range(3):
        proc.stdin.write(json.dumps({"path": wav_paths[0]}) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()

    # Benchmark: 10 calls
    results = []
    for i in range(10):
        path = wav_paths[i % len(wav_paths)]

        t0 = time.perf_counter()
        proc.stdin.write(json.dumps({"path": path}) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        end_to_end_ms = (time.perf_counter() - t0) * 1000

        resp = json.loads(line)
        worker_ms = resp.get("total_ms", 0)
        overhead_ms = end_to_end_ms - worker_ms

        results.append({
            "file": Path(path).name,
            "end_to_end_ms": end_to_end_ms,
            "worker_ms": worker_ms,
            "overhead_ms": overhead_ms,
        })

        print(f"  [{i+1}/10] {Path(path).name}: "
              f"endToEnd={end_to_end_ms:.1f}ms "
              f"worker={worker_ms:.1f}ms "
              f"overhead={overhead_ms:+.1f}ms", file=sys.stderr)

    # Exit worker cleanly
    proc.stdin.write("EXIT\n")
    proc.stdin.flush()
    proc.wait(timeout=5)

    # Summary
    avg_e2e = sum(r["end_to_end_ms"] for r in results) / len(results)
    avg_worker = sum(r["worker_ms"] for r in results) / len(results)
    avg_overhead = sum(r["overhead_ms"] for r in results) / len(results)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"RESULTS (10 calls, after warmup)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Avg end-to-end:   {avg_e2e:.1f}ms", file=sys.stderr)
    print(f"Avg worker-internal: {avg_worker:.1f}ms", file=sys.stderr)
    print(f"Avg IPC overhead: {avg_overhead:+.1f}ms", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"IPC overhead = JSON serialize + pipe write + pipe read + JSON parse", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"Key savings vs cold-start Python:", file=sys.stderr)
    print(f"  - No model reload (3.2s saved)", file=sys.stderr)
    print(f"  - Per-call IPC overhead is just {avg_overhead:.1f}ms", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"In voice-engine integration:", file=sys.stderr)
    print(f"  - Audio loading via Swift AVFoundation (~5ms)", file=sys.stderr)
    print(f"  - IPC to worker: ~{avg_overhead:.1f}ms", file=sys.stderr)
    print(f"  - Worker inference: ~{avg_worker:.1f}ms", file=sys.stderr)
    print(f"  - Total estimated: ~{avg_worker + avg_overhead + 5:.1f}ms", file=sys.stderr)


if __name__ == "__main__":
    main()
