import AppKit
import Foundation
import VoiceEngine

// Simple flag parser: extracts --key value pairs from arguments.
func parseFlags(_ args: [String]) -> [String: String] {
    var flags: [String: String] = [:]
    var i = 1
    while i < args.count {
        if args[i].hasPrefix("--") {
            let key = String(args[i].dropFirst(2))
            if i + 1 < args.count, !args[i + 1].hasPrefix("--") {
                flags[key] = args[i + 1]
                i += 2
            } else {
                flags[key] = "true"
                i += 1
            }
        } else {
            i += 1
        }
    }
    return flags
}

// Entry point
let args = CommandLine.arguments
let flags = parseFlags(args)

if let path = flags["file"] {
    let modelDir = flags["model-dir"]
    Bench.bench(file: path, modelDir: modelDir)
    exit(0)
} else if let corpus = flags["bench"] {
    let modelDir = flags["model-dir"]
    let outputDir = flags["output-dir"] ?? "bench_data/results"
    Bench.benchCorpus(modelDir: modelDir, corpus: corpus, outputDir: outputDir)
    exit(0)
} else if let text = flags["paste"] {
    // Paste audit mode: set clipboard + post Cmd+V
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    Paster.paste(text)
    // Small delay to let the CGEvent flush before exit
    Thread.sleep(forTimeInterval: 0.1)
    exit(0)
} else if args.count >= 3, args[1] == "--file" {
    // Backward-compatible positional form
    let modelDir = flags["model-dir"]
    Bench.bench(file: args[2], modelDir: modelDir)
    exit(0)
} else if args.count >= 3, args[1] == "--paste" {
    let text = args[2]
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    Paster.paste(text)
    Thread.sleep(forTimeInterval: 0.1)
    exit(0)
} else {
    let delegate = AppDelegate()
    NSApplication.shared.delegate = delegate
    NSApplication.shared.run()
}
