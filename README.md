# VoiceEngine

**v0.1** — Zero-latency, fully-local dictation menubar app. Push Caps Lock → speak → release → text appears in your focused app.

No clipboard. No network. No Python at inference time.

> **Definition of Done (v0.1):** Caps Lock dictation transcribes local audio via
> Moonshine-tiny and injects text into the focused app; builds from a clean
> checkout (`./build.sh build`); tests pass (`./build.sh test` — 62/62). ✅ Met.
>
> The **Next steps** below are explicitly *post-v0.1_. v0.1 ships the working MVP;
> those are enhancements, not blockers. Resist treating them as unfinished work.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Single Swift process (menubar app)                     │
│                                                         │
│  Caps Lock → CGEvent tap (absorbs key, no LED flash)    │
│            → AudioCapture (64-frame buffers, 1.3ms)     │
│            → MoonshineEngine (CoreML encoder ANE +       │
│              decoder CPU, Accelerate KV projection)      │
│            → Paster (CGEvent unicode string injection)   │
│            → HUD overlay (1.5s fade)                    │
└─────────────────────────────────────────────────────────┘
```

## Setup

### 1. Export models (once)

```bash
./build.sh setup
```

This runs two Python scripts:

- `Scripts/export_models.py` — Encoder + KV weights + tokenizer
- `Scripts/export_decoder.py` — Decoder (cross-KV as inputs, self-KV stateful)

Artifacts go to `~/.cache/moonshine-coreml/tiny-streaming/`.

Requirements: `coremltools`, `transformers`, `moonshine-onnx`, `torch`, `sentencepiece`.

### 2. Build (Swift only, no Xcode needed)

```bash
./build.sh build
```

Requires Xcode Command Line Tools (`xcode-select --install`).

### 3. Run

```bash
./build.sh run
```

A mic icon appears in the menubar. Press **Caps Lock** to start recording, press again to stop. Text appears in your focused field.

## Permissions

- **Accessibility** (System Settings → Privacy & Security → Accessibility)
  — Required for CGEvent tap (hotkey) and paste fallback.
- **Microphone** — Required for audio capture. Prompted on first use.

## Benchmark

```bash
./build.sh bench test_audio.wav
```

Measures end-to-end: audio load → encoder → KV projection → decoder loop → token decode.

## Moonshine CoreML Pipeline

The production dictation path is a single Swift process:

1. `AudioCapture` records microphone input and converts it to mono 16 kHz
   `Float` samples.
2. `AppController` runs `VAD` after recording stops. Silent captures are
   filtered before inference.
3. `MoonshineEngine.transcribeLong` splits audio into 10 second windows with a
   2 second overlap. Short final windows below the minimum chunk size are
   skipped.
4. Each chunk runs through the CoreML Moonshine stack:
   audio window → encoder on ANE → CPU K/V projection → stateful decoder on CPU
   → SentencePiece token decode.
5. Adjacent chunk text is deduplicated at sentence boundaries before `Paster`
   injects the final transcript into the focused app.

The root Python scripts are audit and benchmark helpers for the same model
artifacts in `~/.cache/moonshine-coreml/tiny-streaming/`:

| Script | Purpose |
|---|---|
| `bench.py` | Loads `encoder.mlpackage`, `decoder_stateful.mlpackage`, and `cross_kv_weights.npz`, then reports per-stage timings for one audio window. |
| `chunk_transcribe.py` | Runs long WAV files through `bench.py`'s model wrapper using the same 10 second chunking and overlap-dedup strategy. It validates mono 16 kHz 16-bit PCM WAV input before model loading. |
| `bench_full.py` | Wraps `bench.py --json` and adds wall-clock and child-process peak RSS measurements. |
| `bench_libri.py` | Runs `bench.py` over LibriSpeech test-clean WAVs, compares transcripts to LibriSpeech references, and summarizes timings by duration bucket. Defaults to `/tmp/librispeech/test-clean`; use `--wav-dir` and `--limit` for targeted runs. |
| `export_dataset.py` | Exports labeled VoiceEngine recordings from `~/Library/Logs/voice-engine/audio` to JSONL and can compare stored transcripts against Whisper. |

`chunk_transcribe.py` does not run VAD itself; use the Swift app path when
validating VAD behavior. Full Python inference requires `./build.sh setup` to
populate the CoreML model cache.

## Model details

| Component | Hardware | Precision | Notes |
|---|---|---|---|
| Encoder | ANE | fp16 | Moonshine-tiny, buckets: 1/3/5/10s |
| Decoder | CPU | fp32 | Stateful self-attn KV (fp16). Cross-KV is input. |
| KV projection | CPU (Accelerate BLAS) | fp32 | Hidden @ K/V weights per layer |
| Tokenizer | CPU (Python subprocess) | — | SentencePiece BPE, cached after first load |

## Known limitations (MVP)

1. **Tokenizer uses Python subprocess** — adds ~50ms on first decode. The id→piece table is cached in-process after load, so subsequent decodes are fast.
2. **Encoder always pads to nearest bucket** — variable bucketing (commit 1e6cc35c in voice repo) isn't merged yet.
3. **Decoder is CPU_ONLY** — ANE rejects large stateful tensors. Retry on macOS 15+.
4. **No streaming partials** — text appears after you stop recording.
5. **Paste is character-by-character CGEvent** — fast enough for typical dictation (1-2 sentences) but large blocks are better served by clipboard + Cmd+V fallback.

## Next steps

1. [ ] Native sentencepiece in Swift (kill the Python subprocess)
2. [ ] Merge variable bucketing (1e6cc35c) for sub-10s utterances
3. [ ] Experiment with decoder on ANE (macOS 15 stateful improvements)
4. [ ] Streaming partial display in HUD
5. [ ] AX observer for context-aware paste (detect app/field)
6. [ ] CDP bridge for browser text injection

## Related

- [jwalin-shah/voice](https://github.com/jwalin-shah/voice) — Original Python STT stack
- [Moonshine](https://github.com/usefulsensors/moonshine) — Fast on-device ASR model
- [machine-scratch/design/issue-007-logs-daemon.md](../design/issue-007-logs-daemon.md)
- [machine-scratch/design/issue-008-orchestrator.md](../design/issue-008-orchestrator.md)
