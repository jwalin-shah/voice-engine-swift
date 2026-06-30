import AppKit
import AVFoundation

@MainActor
public final class AppController {
    /// Result from the transcription + command pipeline.
    private struct TranscriptResult: Sendable {
        let text: String
        let deferredCommand: CommandParser.VoiceCommand?
    }

    public enum State { case idle, recording, transcribing }

    private let statusItem: NSStatusItem
    private let capture = AudioCapture()
    private var hotkey: HotkeyMonitor?
    private let hud = HUDWindow()
    private nonisolated let engine = MoonshineEngine()

    public private(set) var state: State = .idle { didSet { updateMenu() } }
    private var engineLoaded = false
    private let settingsWindow = SettingsWindow()
    private let daemonService = DaemonService()
    private lazy var cleanupService = CleanupService(daemon: daemonService)

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
                let commandResult: TranscriptResult = try await Task.detached(priority: .userInitiated) { [engine, bundleID, cs] in
                    let startTime = CFAbsoluteTimeGetCurrent()
                    let rawFloats = buffer.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                    var text = try engine.transcribeLong(rawAudio: rawFloats)
                    if cs.mode != .disabled {
                        text = await cs.clean(text)
                    }
                    // Pure command detection
                    if let command = CommandParser.parse(text) {
                        let ok = CommandParser.execute(command)
                        NSLog("[VoiceEngine] command executed: \(ok) - \(command)")
                        return TranscriptResult(text: "", deferredCommand: nil)
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
                    return TranscriptResult(text: rawText, deferredCommand: deferredCmd)
                }.value
                let textToPaste = commandResult.text
                let deferredCommand = commandResult.deferredCommand
                guard !textToPaste.isEmpty else { state = .idle; return }
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
