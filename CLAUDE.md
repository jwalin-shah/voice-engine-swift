# voice-engine-swift — Local Dictation Engine

## Overview

Zero-latency, fully-local macOS dictation menubar app. Right Shift (via Karabiner) triggers mic recording; Moonshine-tiny (CoreML on ANE/CPU) transcribes speech and injects text into the focused app. No clipboard, no network, no Python at inference time.

## Architecture

- Swift Package Manager project
- CoreML model: Moonshine-tiny (Apple Neural Engine)
- Menubar app with recording indicator
- Karabiner integration for hotkey trigger
- Daemon for background persistence

## Build

```bash
swift build
swift run
```

## Related

- Karabiner profile "Orbit" maps Right Cmd (tap) to voice toggle
- Binary installed at ~/.local/bin/voice-engine
- Part of the jw ecosystem
