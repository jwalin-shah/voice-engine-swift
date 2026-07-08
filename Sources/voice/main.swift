import AppKit
import Foundation
import VoiceEngine

// Entry point
if CommandLine.arguments.count >= 3, CommandLine.arguments[1] == "--file" {
    Bench.bench(file: CommandLine.arguments[2])
    exit(0)
} else {
    let delegate = AppDelegate()
    NSApplication.shared.delegate = delegate
    NSApplication.shared.run()
}
