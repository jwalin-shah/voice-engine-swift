# Voice Engine — Wayfinder Map

Label: `wayfinder:map`
Created: 2026-07-17

## Destination

A macOS dictation menubar app that transcribes speech to text with local
CoreML models, punctuates with MLX FullStop, and corrects vocabulary with
deterministic rules. Zero cloud dependency. Runs as a menubar app triggered
via Karabiner (Right Cmd tap).

## Current State

- **Branch:** `main` (wayfinder branch `wayfinder/voice-wip-20260714` merged)
- **Deployed:** Yes — running as PID 4736, LaunchAgent managed
- **Build:** ✅ `swift build` passes
- **Tests:** 348/349 pass
  - 1 pre-existing failure: `MoonshineEngine sequential transcribe` — CoreML error code -14 (model architecture vs macOS version)
- **Models:** Moonshine (transcription), FullStop (punctuation), ParakeetMLX (TTS — weights in repo)
- **Services:** CapitalizationService, PunctuationService, VocabularyService, CleanupService, VAD, CommandParser

## Recent work

- Wayfinder branch merged to main
- Branch `wayfinder/voice-wip-20260714` deleted
- `1bfbb6e` — CapitalizationService (sentence-start + proper noun dict + pronoun i)
- `df247fa` — PunctuationService restored as actor, tokenizer padding removed
- `65f6e90` — FullStop punctuation model integrated (#19)
- `ff065f7` — gitignore venv, bench data, model weights
- Today: VocabularyService substring-match bug fixed (regex → manual character boundaries)

## Tickets

### 🔴 Active

1. **Fix CoreML error -14** — MoonshineEngine test failure. Likely a model version incompatibility with current macOS/CoreML. Either update model or pin CoreML compute unit.

2. **Complete adversarial test suite** — BRIEF.md specifies 4 areas: VAD boundary attacks, CommandParser mutation attacks, MoonshineEngine.chunkRanges edge cases, VocabularyService mutation attacks. Partially complete.

3. **Remove MLStateDebug.swift** — dead code identified in dead-code review. Zero callers. Safe to delete.

### 🟡 Next

4. **ParakeetMLX integration** — TTS model weights exist but synthesis path is not wired into the app. Is this in scope?

5. **Voice commands beyond dictation** — CommandParser handles undo, new-line, select, delete. What other commands should it support? What's the UX model?

### 🔵 Future

6. **Test coverage for past bugs** — each fixed bug should have a regression test. Check git log for past fixes and verify coverage.

7. **Benchmark suite** — transcription latency, punctuation accuracy, memory usage. Already has `bench-data/` directory.

## Not yet specified

- **ParakeetMLX scope:** Was TTS always planned, or is this vestigial?
- **Dictation quality metrics:** How do we measure if transcription is improving?
- **App distribution:** Is this just for the captain's machine, or meant to be shareable?
