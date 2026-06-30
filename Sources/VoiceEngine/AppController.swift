import AppKit
import AVFoundation

@MainActor
public final class AppController {
    /// Result from the transcription + command pipeline.
    private struct TranscriptResult: Sendable {
        let text: String
        let deferredCommand: CommandParser.VoiceCommand?
        var transcriptionMs: Double = 0
        var audioSecs: Double = 0
    }

    public enum State { case idle, recording, transcribing }

    private let statusItem: NSStatusItem
    private let capture = AudioCapture()
    private var hotkey: HotkeyMonitor?
    private let hud = HUDWindow()
    private nonisolated let engine = MoonshineEngine()
    private let vad = VAD()
    private let skipTranscriptionIfSilent = true

    public private(set) var state: State = .idle { didSet { updateMenu() } }
    private var engineLoaded = false
    private let settingsWindow = SettingsWindow()
    private let daemonService = DaemonService()
    private lazy var cleanupService = CleanupService(daemon: daemonService)

    private static let logDir: URL = {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/voice-engine")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: dir.appendingPathComponent("audio"), withIntermediateDirectories: true)
        return dir
    }()
    private static var metricsURL: URL { logDir.appendingPathComponent("metrics.jsonl") }

    public init() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    }

    public func run() {
        NSApp.setActivationPolicy(.accessory)
        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "mic.fill", accessibilityDescription: "Voice")
            button.imagePosition = .imageLeft
        }
        updateMenu()
        installHotkey()
        preloadEngine()
    }

    private func updateMenu() {
        let menu = NSMenu()
        let header = NSMenuItem(title: stateTitle(), action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)
        menu.addItem(.separator())
        let engineStatus = NSMenuItem(
            title: engineLoaded ? "Moonshine CoreML" : "Moonshine CoreML loading...",
            action: nil, keyEquivalent: "")
        engineStatus.isEnabled = false
        menu.addItem(engineStatus)
        let builtInMic = NSMenuItem(
            title: "Always use Built-in Mic",
            action: #selector(toggleBuiltInMic), keyEquivalent: "")
        builtInMic.target = self
        builtInMic.state = capture.forceBuiltInMic ? .on : .off
        menu.addItem(builtInMic)
        menu.addItem(.separator())
        let settings = NSMenuItem(title: "Settings...", action: #selector(openSettings), keyEquivalent: ",")
        settings.target = self
        menu.addItem(settings)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "Quit VoiceEngine", action: #selector(quit), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)
        statusItem.menu = menu
    }

    private func stateTitle() -> String {
        switch state {
        case .idle:
            return engineLoaded
                ? "VoiceEngine Press Caps Lock to dictate"
                : "VoiceEngine Loading models..."
        case .recording: return "VoiceEngine Recording..."
        case .transcribing: return "VoiceEngine Transcribing..."
        }
    }

    @objc private func toggleBuiltInMic(_ sender: NSMenuItem) {
        capture.forceBuiltInMic.toggle()
        sender.state = capture.forceBuiltInMic ? .on : .off
    }

    @objc private func openSettings() { settingsWindow.show() }
    @objc private func quit() { NSApp.terminate(nil) }

    private func installHotkey() {
        let monitor = HotkeyMonitor { [weak self] in self?.handleHotkey() }
        do {
            try monitor.start()
            hotkey = monitor
        } catch {
            presentError(error.localizedDescription)
        }
    }

    private func handleHotkey() {
        if state == .transcribing { return }
        if state == .recording { finishRecording() }
        else { beginRecording() }
    }

    private func beginRecording() {
        guard state == .idle else { return }
        state = .recording
        let eng = engine
        capture.partialCallback = { [weak hud] audioData in
            let rawFloats = audioData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            guard rawFloats.count >= 16000 else { return }
            Task.detached(priority: .background) {
                let text = try? eng.transcribeLong(rawAudio: rawFloats)
                let trimmed = text?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                if !trimmed.isEmpty {
                    await MainActor.run { hud?.show(trimmed) }
                }
            }
        }
        do {
            try capture.start()
        } catch {
            state = .idle
            capture.partialCallback = nil
            presentError("Audio start failed: \(error.localizedDescription)")
        }
    }

    private func finishRecording() {
        guard state == .recording else { return }
        capture.partialCallback = nil
        capture.stop()
        guard let buffer = capture.takeBuffer() else { state = .idle; return }
        state = .transcribing
        let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? ""
        Task {
            do {
                let cs = cleanupService
                let rawFloats = buffer.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                // VAD: skip transcription if audio is silent (accidental hotkey press)
                if skipTranscriptionIfSilent && !vad.isSpeech(rawFloats) {
                    NSLog("[VoiceEngine] VAD filtered silent audio (\(rawFloats.count) samples)")
                    writeMetric(["event": "vad_filtered", "samples": rawFloats.count])
                    state = .idle
                    return
                }
                // Save audio WAV before transcription (sidecar gets updated with text later)
                let audioFile = saveAudio(floats: rawFloats, transcription: nil, app: bundleID)
                let commandResult: TranscriptResult = try await Task.detached(priority: .userInitiated) { [engine, bundleID, cs, rawFloats] in
                    let startTime = CFAbsoluteTimeGetCurrent()
                    let audioSecs = Double(rawFloats.count) / 16000.0
                    var text = try engine.transcribeLong(rawAudio: rawFloats)
                    if cs.mode != .disabled {
                        text = await cs.clean(text)
                    }
                    // Pure command detection
                    if let command = CommandParser.parse(text) {
                        let ok = CommandParser.execute(command)
                        NSLog("[VoiceEngine] command executed: \(ok) - \(command)")
                        return TranscriptResult(text: "", deferredCommand: nil, transcriptionMs: (CFAbsoluteTimeGetCurrent() - startTime) * 1000, audioSecs: audioSecs)
                    }
                    // Suffix command
                    let rawText: String
                    let deferredCmd: CommandParser.VoiceCommand?
                    if let (prefix, command) = CommandParser.extractCommand(from: text) {
                        let withVocab = VocabularyService.shared.process(prefix, frontAppBundleID: bundleID)
                        rawText = withVocab.trimmingCharacters(in: .whitespacesAndNewlines)
                        deferredCmd = command
                    } else {
                        let withVocab = VocabularyService.shared.process(text, frontAppBundleID: bundleID)
                        rawText = withVocab.trimmingCharacters(in: .whitespacesAndNewlines)
                        deferredCmd = nil
                    }
                    let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
                    NSLog("[VoiceEngine] transcription: \(String(format: "%.1f", elapsed)) ms, \(rawText.count) chars")
                    return TranscriptResult(text: rawText, deferredCommand: deferredCmd, transcriptionMs: elapsed, audioSecs: audioSecs)
                }.value
                let textToPaste = commandResult.text
                let deferredCommand = commandResult.deferredCommand
                guard !textToPaste.isEmpty else { state = .idle; return }

                // Update JSON sidecar with transcription text (dataset pairing)
                if let audioPath = audioFile {
                    let jsonURL = URL(fileURLWithPath: audioPath).deletingPathExtension().appendingPathExtension("json")
                    if var meta = try? JSONSerialization.jsonObject(with: Data(contentsOf: jsonURL)) as? [String: Any] {
                        meta["text"] = textToPaste
                        meta["transcription_ms"] = commandResult.transcriptionMs
                        meta["audio_secs"] = commandResult.audioSecs
                        if let updated = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .withoutEscapingSlashes]) { try? updated.write(to: jsonURL) }
                    }
                }

                writeMetric(["event": "transcription", "is_command": deferredCommand != nil,
                             "transcription_ms": commandResult.transcriptionMs,
                             "audio_secs": commandResult.audioSecs,
                             "chars": textToPaste.count,
                             "words": textToPaste.split(separator: " ").count,
                             "app": bundleID,
                             "audio_file": audioFile as Any])

                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(textToPaste, forType: .string)
                let logPath = FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent("voice-history.txt")
                let fmt = DateFormatter()
                fmt.dateFormat = "yyyy-MM-dd HH:mm:ss"
                let logLine = "\(fmt.string(from: Date())) | \(textToPaste)\n"
                if let handle = try? FileHandle(forWritingTo: logPath) {
                    handle.seekToEndOfFile()
                    if let data = logLine.data(using: .utf8) { handle.write(data) }
                    handle.closeFile()
                } else {
                    try? logLine.write(to: logPath, atomically: true, encoding: .utf8)
                }
                Paster.paste(textToPaste)
                // Execute deferred command after paste (suffix commands)
                if let command = deferredCommand {
                    CommandParser.execute(command)
                }
            } catch {
                presentError(error.localizedDescription)
            }
            state = .idle
        }
    }

    private func preloadEngine() {
        Task {
            do {
                try await Task.detached(priority: .utility) { [engine] in try engine.load() }.value
                engineLoaded = true
                updateMenu()
                NSLog("[VoiceEngine] Moonshine CoreML engine loaded")
            } catch {
                NSLog(" Engine load failed: \(error.localizedDescription)")
                presentError("Engine load failed: \(error.localizedDescription)")
            }
        }
        Task {
            do {
                try await daemonService.launch()
                let avail = await cleanupService.checkAvailability()
                NSLog("[VoiceEngine] Cleanup daemon ready, model_loaded=\(avail)")
            } catch {
                NSLog("[VoiceEngine] Cleanup daemon unavailable: \(error.localizedDescription)")
            }
        }
    }

    /// Save float32 audio as a 32-bit float WAV file for later analysis.
    private func saveAudio(floats: [Float], transcription: String?, app: String?) -> String? {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd-HH-mm-ss"
        let name = fmt.string(from: Date()) + ".wav"
        let url = Self.logDir.appendingPathComponent("audio/\(name)")
        guard writeWAV(floats: floats, to: url) else { return nil }
        // Save JSON sidecar with metadata (transcription, app, etc.) for dataset building
        var meta: [String: Any] = ["ts": fmt.string(from: Date()), "duration_secs": Double(floats.count) / 16000.0]
        if let t = transcription { meta["text"] = t }
        if let a = app { meta["app"] = a }
        if let jsonData = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .withoutEscapingSlashes]) {
            let jsonURL = url.deletingPathExtension().appendingPathExtension("json")
            try? jsonData.write(to: jsonURL)
        }
        return url.path
    }

    private func writeWAV(floats: [Float], to url: URL) -> Bool {
        let sampleRate: UInt32 = 16000
        let channels: UInt16 = 1
        let bitsPerSample: UInt16 = 32
        let dataSize = UInt32(floats.count * 4)
        var wav = Data()
        func append<T: FixedWidthInteger>(_ v: T) { withUnsafeBytes(of: v.littleEndian) { wav.append(contentsOf: $0) } }
        wav.append(contentsOf: "RIFF".utf8)
        append(UInt32(36 + dataSize))
        wav.append(contentsOf: "WAVEfmt ".utf8)
        append(UInt32(16))
        append(UInt16(3))           // IEEE float
        append(channels)
        append(sampleRate)
        append(sampleRate * 4)
        append(UInt16(4))
        append(bitsPerSample)
        wav.append(contentsOf: "data".utf8)
        append(dataSize)
        floats.withUnsafeBytes { wav.append(contentsOf: $0) }
        return (try? wav.write(to: url)) != nil
    }

    private func writeMetric(_ fields: [String: Any]) {
        var entry = fields
        entry["ts"] = ISO8601DateFormatter().string(from: Date())
        guard let data = try? JSONSerialization.data(withJSONObject: entry),
              let line = String(data: data, encoding: .utf8) else { return }
        let bytes = (line + "\n").data(using: .utf8)!
        if let handle = try? FileHandle(forWritingTo: Self.metricsURL) {
            handle.seekToEndOfFile()
            handle.write(bytes)
            handle.closeFile()
        } else {
            try? bytes.write(to: Self.metricsURL)
        }
    }

    private func presentError(_ message: String) {
        NSLog(" VoiceEngine: \(message)")
        let alert = NSAlert()
        alert.messageText = "VoiceEngine"
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }

}
