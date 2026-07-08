import AppKit
import Foundation
import VoiceEngine

// Trigger Accessibility prompt on first launch.
// Once granted (one click in the system dialog), CGEvent posting and
// AX paste work instantly — no osascript, no private APIs, no hacks.
// Grant persists until the binary is rebuilt (ad-hoc signing changes identity).
if !AXIsProcessTrusted() {
    let opts = [kAXTrustedCheckOptionPrompt.takeRetainedValue() as String: true] as CFDictionary
    AXIsProcessTrustedWithOptions(opts)
}

// Entry point
if CommandLine.arguments.count >= 3, CommandLine.arguments[1] == "--file" {
    Bench.bench(file: CommandLine.arguments[2])
    exit(0)
} else {
    let delegate = AppDelegate()
    NSApplication.shared.delegate = delegate
    NSApplication.shared.run()
}
