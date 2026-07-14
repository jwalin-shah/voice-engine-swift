import AppKit
import AVFoundation

@MainActor
public final class AppController {
    /// Result from the transcription + command pipeline.
    private struct TranscriptResult: Sendable {
        let text: String
        let immediateCommand: CommandParser.VoiceCommand?
        let deferredCommand: CommandParser.VoiceCommand?
        var transcriptionMs: Double = 0
        var audioSecs: Double = 0
    }

    public enum State { case idle, recording, transcribing }

    private let statusItem: NSStatusItem
    private let capture = AudioCapture()
    // HotkeyMonitor (Caps Lock CGEvent tap) removed — hotkey handled via Karabiner → SIGUSR1
    private nonisolated let engine = MoonshineEngine()
    private let vad = VAD()
    private let skipTranscriptionIfSilent = true

    public private(set) var state: State = .idle {
        didSet {
            updateMenu()
            announceStateChange(from: oldValue, to: state)
        }
    }
    private var engineLoaded = false
    private let settingsWindow = SettingsWindow()
    private let cleanupService = CleanupService()
    private var recordingTimer: Timer?
    private var signalSource: DispatchSourceSignal?
    private nonisolated static let maxRecordingSeconds: TimeInterval = 60
    private var transcriptionTask: Task<Void, Never>?
    private var transcriptionTaskID: UUID?

    private nonisolated static let logDir: URL = {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/voice-engine")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: dir.appendingPathComponent("audio"), withIntermediateDirectories: true)
        return dir
    }()
    private nonisolated static var metricsURL: URL { logDir.appendingPathComponent("metrics.jsonl") }

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
        installSignalToggle()
        capture.warmUp()
        preloadEngine()
    }

    private func installSignalToggle() {
        // Allow Karabiner (or any external tool) to toggle recording by sending SIGUSR1.
        signal(SIGUSR1, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: SIGUSR1, queue: .main)
        source.setEventHandler { [weak self] in
            NSLog("[VoiceEngine] SIGUSR1 received")
            self?.handleHotkey()
        }
        source.resume()
        signalSource = source  // retain so it isn't deallocated
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
                ? "VoiceEngine Right Shift to dictate"
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

    private func handleHotkey() {
        Foundation.NSLog("[AppController] handleHotkey called, state=\(state)")
        switch state {
        case .transcribing:
            transcriptionTask?.cancel()
            transcriptionTask = nil
            transcriptionTaskID = nil
            state = .idle
            Foundation.NSLog("[AppController] cancelled transcription from hotkey")
        case .recording:
            finishRecording()
        case .idle:
            beginRecording()
        }
    }

    private func beginRecording() {
        Foundation.NSLog("[AppController] beginRecording")
        guard state == .idle else { return }
        state = .recording
        recordingTimer?.invalidate()
        let timeout = Self.maxRecordingSeconds
        recordingTimer = Timer.scheduledTimer(withTimeInterval: timeout, repeats: false) { [weak self] _ in
            Task { @MainActor in
                NSLog("[VoiceEngine] Recording timeout after \(timeout)s, stopping")
                self?.finishRecording()
            }
        }
        do {
            try capture.start()
            Foundation.NSLog("[AppController] capture.start succeeded")
        } catch {
            recordingTimer?.invalidate()
            recordingTimer = nil
            state = .idle
            capture.partialCallback = nil
            Foundation.NSLog("[AppController] capture.start failed: \(error.localizedDescription)")
            presentError("Audio start failed: \(error.localizedDescription)")
        }
    }

    private func finishRecording() {
        Foundation.NSLog("[AppController] finishRecording")
        guard state == .recording else { return }
        recordingTimer?.invalidate()
        recordingTimer = nil
        capture.partialCallback = nil
        capture.stop()
        guard let buffer = capture.takeBuffer() else { state = .idle; return }
        state = .transcribing
        let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? ""
        let taskID = UUID()
        transcriptionTaskID = taskID
        transcriptionTask = Task {
            defer {
                if transcriptionTaskID == taskID {
                    transcriptionTask = nil
                    transcriptionTaskID = nil
                    state = .idle
                }
            }
            do {
                let cs = cleanupService
                let rawFloats = buffer.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                try Task.checkCancellation()
                // VAD: skip transcription if audio is silent (accidental hotkey press)
                if skipTranscriptionIfSilent && !vad.isSpeech(rawFloats) {
                    NSLog("[VoiceEngine] VAD filtered silent audio (\(rawFloats.count) samples)")
                    Self.writeMetric(["event": "vad_filtered", "samples": rawFloats.count])
                    return
                }
                // Save audio WAV before transcription (sidecar gets updated with text later)
                let audioFile = saveAudio(floats: rawFloats, transcription: nil, app: bundleID)
                let commandResult: TranscriptResult = try await Task.detached(priority: .userInitiated) { [engine, bundleID, cs, rawFloats] in
                    let startTime = CFAbsoluteTimeGetCurrent()
                    let audioSecs = Double(rawFloats.count) / 16000.0
                    var timing: TranscribeTiming? = nil
                    var text = try engine.transcribeLong(rawAudio: rawFloats, timing: &timing)
                    if cs.mode != .disabled {
                        text = await cs.clean(text)
                    }
                    // Pure command detection
                    if let command = CommandParser.parse(text) {
                        return TranscriptResult(text: "", immediateCommand: command, deferredCommand: nil, transcriptionMs: (CFAbsoluteTimeGetCurrent() - startTime) * 1000, audioSecs: audioSecs)
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
                    return TranscriptResult(text: rawText, immediateCommand: nil, deferredCommand: deferredCmd, transcriptionMs: elapsed, audioSecs: audioSecs)
                }.value
                try Task.checkCancellation()
                if let command = commandResult.immediateCommand {
                    let ok = CommandParser.execute(command)
                    NSLog("[VoiceEngine] command executed: \(ok) - \(command)")
                    return
                }
                let textToPaste = commandResult.text
                let deferredCommand = commandResult.deferredCommand
                guard !textToPaste.isEmpty else { return }

                // Paste immediately — file I/O happens after, never blocks the user.
                if Paster.paste(textToPaste) {
                    signalPasteCompleted()
                }
                // Execute deferred command after paste (suffix commands)
                if let command = deferredCommand {
                    CommandParser.execute(command)
                }

                // Fire-and-forget: save training data, metrics, history in background.
                let audioPath = audioFile
                let cmdMs = commandResult.transcriptionMs
                let cmdAudioSecs = commandResult.audioSecs
                let hasDeferred = deferredCommand != nil
                Task.detached(priority: .background) {
                    // Update JSON sidecar with transcription text (dataset pairing)
                    if let path = audioPath {
                        let jsonURL = URL(fileURLWithPath: path).deletingPathExtension().appendingPathExtension("json")
                        if var meta = try? JSONSerialization.jsonObject(with: Data(contentsOf: jsonURL)) as? [String: Any] {
                            meta["text"] = textToPaste
                            meta["transcription_ms"] = cmdMs
                            meta["audio_secs"] = cmdAudioSecs
                            if let updated = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .withoutEscapingSlashes]) { try? updated.write(to: jsonURL) }
                        }
                    }

                    Self.writeMetric(["event": "transcription", "is_command": hasDeferred,
                                 "transcription_ms": cmdMs,
                                 "audio_secs": cmdAudioSecs,
                                 "chars": textToPaste.count,
                                 "words": textToPaste.split(separator: " ").count,
                                 "app": bundleID,
                                 "audio_file": audioPath as Any])

                    let logPath = Self.logDir.appendingPathComponent("voice-history.txt")
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
                }
            } catch is CancellationError {
                NSLog("[VoiceEngine] transcription cancelled")
            } catch {
                presentError(error.localizedDescription)
            }
        }
    }

    private func signalPasteCompleted() {
        NSHapticFeedbackManager.defaultPerformer.perform(.generic, performanceTime: .now)
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

    private nonisolated static func writeMetric(_ fields: [String: Any]) {
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

    private func announceStateChange(from oldState: State, to newState: State) {
        guard oldState != newState else { return }
        let message: String
        switch newState {
        case .idle:       message = "Voice engine idle"
        case .recording:  message = "Voice engine recording"
        case .transcribing: message = "Voice engine transcribing"
        }
        NSAccessibility.post(
            element: NSApp,
            notification: .announcementRequested,
            userInfo: [.announcement: NSAttributedString(string: message)]
        )
        NSLog("[VoiceEngine] accessibility announcement: \(message)")
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
