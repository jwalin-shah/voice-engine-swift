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

1. **Fix CoreML error -14** — 🔵 BLOCKED, root cause confirmed 2026-07-20, not fixable from this repo alone.

   **Correction to earlier note:** the claim "no export script found in-repo" was wrong — the earlier search missed `Scripts/`. The full export pipeline IS in-repo and git-tracked: `Scripts/export_decoder_final.py` (decoder, cross-KV as inputs + self-KV as CoreML `states`), `Scripts/export_cross_kv.py`, `Scripts/export_models.py` (full encoder+decoder+tokenizer pipeline). Model source: `UsefulSensors/moonshine-tiny` on HuggingFace, downloaded live via `transformers`. `.venv/bin/python3.12` is a dead symlink (homebrew `python@3.12` was unlinked/removed) — use `uv venv --python ~/.local/bin/python3.12` + `uv pip install coremltools torch transformers numpy sentencepiece` instead (installs coremltools 9.0 cleanly).

   **Real root cause, proven 2026-07-20:** this is an OS-level CoreML runtime defect on this machine's build (macOS 27.0, build **26A5378n** — looks like a seed/beta build), not anything about the Moonshine model or the export scripts. Evidence:
   - Freshly re-ran `Scripts/export_decoder_final.py` end-to-end (fresh HF download, fresh coremltools 9.0 convert) → the newly-built `decoder_stateful.mlpackage` fails with the **exact same** `error code: -14` immediately inside coremltools' own `MLModel` load, before the file ever reaches the Swift app.
   - Re-tried with `minimum_deployment_target` bumped from `ct.target.iOS18` to `ct.target.iOS26` (coremltools 9.0's newest available target) → same failure.
   - Isolated with a **minimal repro**: a 4-float `torch.nn.Module` with a single `ct.StateType` buffer (no Moonshine code at all) → fails with identical `error code: -14` at execution-plan build time.
   - Same minimal repro **without** any `states=[...]` (plain stateless model, otherwise identical) → builds and runs `predict()` successfully.
   - Tried the stateful repro across `compute_units` = `CPU_ONLY`, `ALL`, `CPU_AND_GPU` → fails identically in all three (rules out compute-unit pinning as a variable, confirming today's earlier finding from a different angle).
   
   **Conclusion:** Core ML's *stateful model* feature (`ct.StateType`, used for the KV-cache-style state Moonshine's decoder needs) appears to be broken at the OS/runtime level on this machine's current macOS 27.0 build, independent of model architecture, target spec, or compute unit. No amount of re-exporting `decoder_stateful` will fix this — the task's original premise (re-export against current SDK) is falsified by direct evidence.
   
   **Decision (captain, 2026-07-20): paused, not rearchitected.** Waiting on a macOS update rather than doing the KV-cache-avoidance rewrite now — the app worked correctly before this OS build broke CoreML's stateful-model feature, so this is treated as an OS regression to wait out, not an app bug to route around. LaunchAgent (`org.nixos.com.jwalinshah.voice-engine`) has been `launchctl bootout`'d (not just killed — KeepAlive won't relaunch it) so it stops burning cycles retrying a load that can't succeed. Re-enable via nix-darwin (see `configuration.nix`) once a newer macOS build is confirmed to fix the CoreML regression, or if the KV-cache rearchitecture (option 2 below) gets greenlit instead.

   **Paths forward if this needs revisiting:**
   1. Check for a macOS update — 26A5378n reads like a seed build; a newer build may fix the CoreML runtime regression.
   2. Rearchitect the decoder to avoid CoreML `states` entirely — pass `self_k`/`self_v` as plain input/output tensors and have Swift manage the KV cache buffer between calls instead of relying on CoreML's built-in state. Requires changes to `Scripts/export_decoder_final.py` (drop `states=`, add self_k/self_v to `inputs`/`outputs`) AND to `MoonshineInfer.swift` (feed the cache back in and read it out every step instead of relying on stateful `MLState`). Not attempted — nontrivial surgery on a live production inference path.
   3. Report to Apple as a CoreML regression if reproducible in Xcode/other CoreML apps.
   
   Cache and model files left exactly as found (`decoder_stateful.mlpackage` restored from backup, byte-identical, confirmed via `diff -rq`) — nothing deleted, so re-enabling later is just a LaunchAgent load away.

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
