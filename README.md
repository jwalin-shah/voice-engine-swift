# VoiceEngine

**v0.1** — Zero-latency, fully-local dictation menubar app. Right Shift (via Karabiner) → speak → Right Shift again → text appears in your focused app.

No clipboard. No network. No Python at inference time.

> **Definition of Done (v0.1):** Right Shift dictation (via Karabiner) transcribes local audio via
> Moonshine-tiny and injects text into the focused app; builds from a clean
> checkout (`./bin/build`); tests pass (`./bin/test` — 62/62). ✅ Met.
>
> The **Next steps** below are explicitly *post-v0.1_. v0.1 ships the working MVP;
> those are enhancements, not blockers. Resist treating them as unfinished work.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Single Swift process (menubar app)                     │
│                                                         │
│  Karabiner Right Shift → SIGUSR1 toggle                │
│            → AudioCapture (64-frame buffers, 1.3ms)     │
│            → MoonshineEngine (CoreML encoder ANE +      │
│              decoder CPU, Accelerate KV projection)      │
│            → Paster (voice-typer keystroke injection)    │
│            → HUD overlay (1.5s fade)                    │
└─────────────────────────────────────────────────────────┘
```

## Setup

### 1. Export models (once)

```bash
./bin/setup
```

This runs two Python scripts:

- `Scripts/export_models.py` — Encoder + KV weights + tokenizer
- `Scripts/export_decoder.py` — Decoder (cross-KV as inputs, self-KV stateful)

Artifacts go to `~/.cache/moonshine-coreml/tiny-streaming/`.

Requirements: `coremltools`, `transformers`, `moonshine-onnx`, `torch`, `sentencepiece`.

### 2. Build and run (Swift only, no Xcode needed)

```bash
./bin/build
./bin/run
```

Requires Xcode Command Line Tools (`xcode-select --install`).

This builds the release binary, installs it to `~/local/bin/voice-engine`,
code-signs it, and starts it in the background.

A mic icon appears in the menubar. Trigger dictation with **Right Shift** via Karabiner to start recording, press again to stop. Text appears in your focused field.

### 3. Karabiner config

Import `karabiner-voice-engine.json` in Karabiner-Elements, or add this rule manually:

```json
{
  "description": "VoiceEngine: Right Shift toggles dictation",
  "manipulators": [
    {
      "from": { "key_code": "right_shift", "modifiers": { "optional": ["any"] } },
      "to": [{ "shell_command": "kill -USR1 $(pgrep -x voice-engine)" }],
      "type": "basic"
    }
  ]
}
```

## Permissions

- **Microphone** — Required for audio capture. Prompted on first use.
- **Karabiner-Elements** — Handles the Right Shift hotkey. Make sure Karabiner has Accessibility permission.

VoiceEngine itself does **not** require Accessibility permission.

## Benchmark

```bash
./.build/release/voice-engine --file test_audio.wav
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
| `chunk_transcribe.py` | Runs long WAV files through `bench.py`'s model wrapper using the same 10 second chunking and overlap-dedup strategy. It validates mono 16 kHz WAV input before model loading, accepting either 16-bit PCM or the Swift app's archived 32-bit float WAVs. |
| `bench_full.py` | Wraps `bench.py --json` and adds wall-clock and child-process peak RSS measurements. |
| `bench_libri.py` | Runs `bench.py` over LibriSpeech test-clean WAVs, compares transcripts to LibriSpeech references, and summarizes timings by duration bucket. Defaults to `/tmp/librispeech/test-clean`; use `--wav-dir` and `--limit` for targeted runs. |
| `export_dataset.py` | Exports labeled VoiceEngine recordings from `~/Library/Logs/voice-engine/audio` to JSONL and can compare stored transcripts against Whisper. |

`chunk_transcribe.py` does not run VAD itself; use the Swift app path when
validating VAD behavior. Full Python inference requires `./bin/setup` to
populate the CoreML model cache.

## Model details

| Component | Hardware | Precision | Notes |
|---|---|---|---|
| Encoder | ANE | fp16 | Moonshine-tiny, padded to 10 s window |
| Decoder | CPU | fp32 | Stateful self-attn KV (fp16). Cross-KV is input. |
| KV projection | CPU (Accelerate BLAS) | fp32 | Hidden @ K/V weights per layer |
| Tokenizer | CPU (JSON id→piece) | — | SentencePiece BPE, loaded once at startup |

## Known limitations (MVP)

1. **Decoder is CPU_ONLY** — ANE rejects large stateful tensors. Retry on macOS 15+.
2. **No streaming partials** — text appears after you stop recording.
3. **Paste uses simulated keystrokes** — fast enough for typical dictation (1–2 sentences).

## Next steps

1. [ ] Native SentencePiece in Swift
2. [ ] Experiment with decoder on ANE (macOS 15 stateful improvements)
3. [ ] Streaming partial display in HUD
4. [ ] AX observer for context-aware paste (detect app/field)
5. [ ] CDP bridge for browser text injection

## Related

- [jwalin-shah/voice](https://github.com/jwalin-shah/voice) — Original Python STT stack
- [Moonshine](https://github.com/usefulsensors/moonshine) — Fast on-device ASR model
- [machine-scratch/design/issue-007-logs-daemon.md](../design/issue-007-logs-daemon.md)
- [machine-scratch/design/issue-008-orchestrator.md](../design/issue-008-orchestrator.md)
