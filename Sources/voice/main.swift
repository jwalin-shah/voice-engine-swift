import AppKit
import Foundation
import VoiceEngine

// Entry point
if CommandLine.arguments.count >= 3, CommandLine.arguments[1] == "--file" {
    let path = CommandLine.arguments[2]
    let sem = DispatchSemaphore(value: 0)
    Task {
        await Bench.bench(file: path)
        sem.signal()
    }
    sem.wait()
    exit(0)
} else {
    let delegate = AppDelegate()
    NSApplication.shared.delegate = delegate
    NSApplication.shared.run()
}
