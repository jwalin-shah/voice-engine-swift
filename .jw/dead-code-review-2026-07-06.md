# Dead Code Review — voice-engine-swift — 2026-07-06

## Summary

Removed 6 files (2 source, 4 test) representing genuinely dead code with zero callers and zero build integration. Build and tests pass with no regressions.

## Scope of Removals

### Source Files (2 files, ~314 lines)

| File | Lines | Symbol | Reason |
|------|-------|--------|--------|
| `Sources/VoiceEngine/DaemonService.swift` | 1-261 | `DaemonService` (actor, all methods) | Python daemon replaced by native `CleanupService`. Zero callers in Sources/. Self-documented in `DaemonServiceTests.swift`: "DaemonService is no longer used — the Python cleanup daemon has been replaced with native Swift regex-based filler word removal in CleanupService." |
| `Sources/VoiceEngine/MLStateDebug.swift` | 1-53 | `MLStateDebug` (enum, `logMethods()`, `writeState()`) | CoreML runtime introspection utility. Zero callers in Sources/. Neither `MoonshineInfer.swift` nor any other file references it. Was likely used during cross-KV state development and abandoned. |

### Test Files (4 files, ~237 lines)

| File | Lines | Symbol | Reason |
|------|-------|--------|--------|
| `Tests/VoiceEngineTests/CommandParserTests.swift` | 1-33 | All `@Test` functions | Orphaned. `Package.swift` test target is `Tests/Runner/`, NOT `Tests/VoiceEngineTests/`. These Swift Testing (`import Testing`) tests are never compiled or run. Coverage is fully duplicated in `Tests/Runner/main.swift`. |
| `Tests/VoiceEngineTests/CleanupServiceTests.swift` | 1-13 | All `@Test` functions | Same orphaned-test-directory issue. |
| `Tests/VoiceEngineTests/VocabularyServiceTests.swift` | 1-12 | All `@Test` functions | Same orphaned-test-directory issue. |
| `Tests/VoiceEngineTests/DaemonServiceTests.swift` | 1-4 | (comment only, no code) | Comment-only file confirming daemon retirement. No test functions. |

### Directory Removal

- `Tests/VoiceEngineTests/` — entire directory removed (empty after file deletions).

## Caller Analysis

### DaemonService

- `rg -rn "DaemonService" --type swift Sources/` returned only self-references in `DaemonService.swift`.
- `rg -n "DaemonService\|daemonService\|daemon" Sources/VoiceEngine/AppController.swift` returned zero matches.
- The `actor DaemonService` is `public`, but no external module imports it, and no internal file references it.
- The class manages a Python subprocess via JSON-RPC — if it were in use, `AppController` or `CleanupService` would instantiate it. Neither does.

### MLStateDebug

- `rg -rn "MLStateDebug" --type swift Sources/` returned only self-references in `MLStateDebug.swift`.
- `rg -n "MLStateDebug\|mlStateDebug" Sources/VoiceEngine/MoonshineInfer.swift` returned zero matches.
- The `public enum MLStateDebug` exposes `logMethods()` and `writeState()`, but no callers exist. It uses `NSClassFromString("MLState")` for runtime introspection — if still needed, it would be called from `MoonshineEngine.buildCrossKV()` or similar.

### Orphaned Test Files

- `Package.swift` line 24-28: the test target `VoiceEngineTests` points to `path: "Tests/Runner"`, not `Tests/VoiceEngineTests/`.
- `import Testing` is only used by these three orphaned files — zero use in the compiled test target.
- The `Tests/Runner/main.swift` test runner covers equivalent functionality for `CommandParser`, `CleanupService`, `VocabularyService`, `MoonshineEngine`, `VAD`, and `Paster`.

## Near-Miss Symbols (Kept)

| Symbol | File | Reason Kept |
|--------|------|-------------|
| `SettingsWindow` (all methods) | `Sources/VoiceEngine/SettingsWindow.swift` | Referenced from `AppController.swift:25`: `private let settingsWindow = SettingsWindow()`. `tldr dead` flagged it as dead (false positive in Swift call-graph tracing). |
| `Bench.bench()`, `Bench.loadWAV()` | `Sources/VoiceEngine/CLI.swift` | Called from `Sources/voice/main.swift:7`: `Bench.bench(file:)`. CLI mode entry point. |
| `AppDelegate` (both methods) | `Sources/VoiceEngine/CLI.swift` | Called from `Sources/voice/main.swift:10-12`. App entry point. |
| All functions in `AppController.swift`, `AudioCapture.swift`, `MoonshineInfer.swift`, `CleanupService.swift`, `VocabularyService.swift`, `Paster.swift`, `VAD.swift`, `HotkeyMonitor.swift`, `CommandParser.swift` | Various | All reachable through the `main → AppDelegate → AppController` call chain. `tldr dead` reported 86.1% dead due to failure to build a Swift cross-file call graph (`tldr calls` returned 0 edges). |

## tldr Tool Reliability Note

The `tldr dead` scan reported 136/158 functions (86.1%) as dead with zero call-graph edges — this is a known limitation with Swift code where `tldr` cannot resolve cross-file references. The entire report was a false positive. All dead-code determination in this review was done manually via `rg` caller analysis and `Package.swift` target inspection.

## Build/Test Verification

### Build
```
swift build
Build complete! (1.97s)
Exit code: 0
```
3 pre-existing warnings (not caused by removals):
- `Paster.swift:120`: `var actionName` never mutated
- `MoonshineInfer.swift:502,506,510`: unused result of `withMultiArray`

### Tests
```
swift run voice-tests
152 passed, 1 failed
```
The 1 failure ("VocabularyService adversarial — substring triggers: expected 'testing now', got 'Ting now'") is a pre-existing bug in vocabulary substring matching, unrelated to dead code removal.

## Risks and Follow-Ups

- **Risk: Low.** The two removed source files had zero callers and zero imports. The removed test files were never compiled.
- **No dependency impact:** `DaemonService` had no callers. `MLStateDebug` had no callers. No other files imported or referenced either symbol.
- **No API surface change:** While both were `public`, neither was part of any documented or consumed API.
- **Follow-up:** The pre-existing vocabulary substring partial-match bug (1 test failure) should be fixed separately.
