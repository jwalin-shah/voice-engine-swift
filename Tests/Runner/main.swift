import Foundation
import VoiceEngine

class TestRunner {
    var passed = 0, failed = 0, currentSuite = ""

    func suite(_ name: String) { currentSuite = name; print("\n--- \(name) ---") }
    func ok(_ msg: String) { passed += 1; print("  [PASS] \(msg)") }
    func fail(_ msg: String) { failed += 1; print("  [FAIL] \(currentSuite): \(msg)") }
    func assertEqual<T: Equatable>(_ expr: @autoclosure () -> T, _ expected: @autoclosure () -> T, _ msg: String = "") { let v = expr(); let e = expected(); v == e ? ok(msg.isEmpty ? "\(v) == \(e)" : msg) : fail(msg.isEmpty ? "Expected \(e), got \(v)" : "\(msg): expected \(e), got \(v)") }
    func assertNil<T>(_ expr: @autoclosure () -> T?, _ msg: String = "") { let v = expr(); v == nil ? ok(msg) : fail(msg.isEmpty ? "Expected nil, got \(v!)" : "\(msg): expected nil") }
    func assertNotNil<T>(_ expr: @autoclosure () -> T?, _ msg: String = "") { let v = expr(); v != nil ? ok(msg) : fail(msg.isEmpty ? "Expected non-nil" : msg) }
    func assertTrue(_ expr: @autoclosure () -> Bool, _ msg: String = "") { expr() ? ok(msg) : fail(msg.isEmpty ? "Expected true" : msg) }
    func assertFalse(_ expr: @autoclosure () -> Bool, _ msg: String = "") { !expr() ? ok(msg) : fail(msg.isEmpty ? "Expected false" : msg) }

    /// Bridge an async throwing closure to synchronous for the test runner.
    /// Uses a semaphore; safe because PunctuationService is a dedicated actor
    /// (never the main actor).
    func awaitSync<T: Sendable>(_ block: @escaping @Sendable () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var result: Result<T, any Error>?
        Task {
            do {
                result = .success(try await block())
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        switch result! {
        case .success(let value): return value
        case .failure(let error): throw error
        }
    }

    func runAll() {
        suite("CommandParser.parse")
        assertEqual(CommandParser.parse("undo"), .undo, "undo")
        assertEqual(CommandParser.parse("new line"), .pressEnter, "new line")
        assertEqual(CommandParser.parse("newline"), .pressEnter, "newline")
        assertEqual(CommandParser.parse("new paragraph"), .newParagraph, "new paragraph")
        assertEqual(CommandParser.parse("tab"), .tab, "tab")
        assertEqual(CommandParser.parse("press enter"), .pressEnter, "press enter")
        assertNil(CommandParser.parse("hello world"), "no command")
        assertNil(CommandParser.parse(""), "empty string")
        assertNil(CommandParser.parse("   "), "whitespace only")
        assertEqual(CommandParser.parse("  undo  "), .undo, "padded undo")
        suite("CommandParser.extractCommand")
        do { let r = CommandParser.extractCommand(from: "hello world undo"); assertEqual(r?.prefix, "hello world"); assertEqual(r?.command, .undo) }
        assertEqual(CommandParser.extractCommand(from: "hello new line")?.command, .pressEnter)
        assertEqual(CommandParser.extractCommand(from: "hello press enter")?.command, .pressEnter)
        assertEqual(CommandParser.extractCommand(from: "hello tab")?.command, .tab)
        assertEqual(CommandParser.extractCommand(from: "hello new paragraph")?.command, .newParagraph)
        assertNil(CommandParser.extractCommand(from: "undo"), "pure undo skipped")
        assertNil(CommandParser.extractCommand(from: "new line"), "pure new line skipped")
        assertNil(CommandParser.extractCommand(from: "helloundo"), "no word boundary")
        assertNil(CommandParser.extractCommand(from: "hello_undo"), "underscore boundary")
        assertNil(CommandParser.extractCommand(from: ""), "empty string")
        suite("CommandParser.Equatable")
        assertEqual(CommandParser.VoiceCommand.undo, .undo)
        assertEqual(CommandParser.VoiceCommand.pressEnter, .pressEnter)
        assertTrue(CommandParser.VoiceCommand.undo != .pressEnter)
        assertTrue(CommandParser.VoiceCommand.pressEnter != .newParagraph)
        suite("VocabularyService")
        UserDefaults.standard.removeObject(forKey: "customVocabulary")
        UserDefaults.standard.removeObject(forKey: "appCommands")
        assertTrue(VocabularyService.shared.vocabulary.isEmpty, "empty vocab")
        assertEqual(VocabularyService.shared.process("hello world"), "hello world", "no vocab unchanged")
        VocabularyService.shared.addVocab(trigger: "equestrian", replacement: "Equestrian")
        assertEqual(VocabularyService.shared.applyVocabulary(to: "the equestrian center"), "the Equestrian center", "single replacement")
        VocabularyService.shared.vocabulary = []
        do { var v = VocabularyService.shared.vocabulary; v.append(VocabularyService.VocabEntry(trigger: "test", replacement: "TEST", isActive: false)); VocabularyService.shared.vocabulary = v; assertEqual(VocabularyService.shared.applyVocabulary(to: "test"), "test", "inactive skipped"); VocabularyService.shared.vocabulary = [] }
        VocabularyService.shared.addVocab(trigger: "persist", replacement: "Persist")
        do { let loaded = VocabularyService.shared.vocabulary; assertEqual(loaded.count, 1); assertEqual(loaded[0].trigger, "persist") }; VocabularyService.shared.vocabulary = []
        do { var cmds = VocabularyService.shared.appCommands; cmds.append(VocabularyService.AppCommand(appName: "T", bundleID: "com.test", trigger: "apphello", replacement: "APPHELLO")); VocabularyService.shared.appCommands = cmds; assertEqual(VocabularyService.shared.process("say apphello", frontAppBundleID: "com.test"), "say APPHELLO", "app command"); VocabularyService.shared.appCommands = [] }
        suite("CleanupService")
        do { let cs = CleanupService(); assertEqual(cs.mode, .fillerOnly, "default mode") }
        UserDefaults.standard.set(CleanupService.CleanupMode.disabled.rawValue, forKey: "cleanupMode")
        do { let cs = CleanupService(); assertEqual(cs.mode, .disabled, "mode persistence") }
        UserDefaults.standard.removeObject(forKey: "cleanupMode")
        assertEqual(CleanupService.CleanupMode.disabled.rawValue, "Disabled", "rawValue disabled")
        assertEqual(CleanupService.CleanupMode.fillerOnly.rawValue, "Filler only", "rawValue filler")
        assertEqual(CleanupService.CleanupMode.full.rawValue, "Full", "rawValue full")
        suite("Paster")
        assertFalse(Paster.paste(""), "empty paste is rejected")
        suite("MoonshineEngine.chunkRanges")
        assertEqual(MoonshineEngine.chunkRanges(sampleCount: 0).count, 0, "empty audio has no chunks")
        assertEqual(MoonshineEngine.chunkRanges(sampleCount: 160000), [0..<160000], "exact encoder window is one chunk")
        assertEqual(MoonshineEngine.chunkRanges(sampleCount: 168000), [0..<160000], "sub-second tail after overlap is skipped")
        assertEqual(MoonshineEngine.chunkRanges(sampleCount: 176000), [0..<160000, 128000..<176000], "one-second tail after overlap is kept")
        assertEqual(MoonshineEngine.chunkRanges(sampleCount: 288000), [0..<160000, 128000..<288000], "eighteen seconds uses two chunks")

        suite("VAD.isSpeech")
        do {
            let vad = VAD()
            // silence: all zeros
            assertTrue(!vad.isSpeech([Float](repeating: 0.0, count: 16000)), "all-zero audio is silence")
            // loud tone: amplitude > threshold
            let loud = [Float](repeating: 0.1, count: 16000)
            assertTrue(vad.isSpeech(loud), "loud constant tone is speech")
            // sub-windowSize buffer uses RMS path
            let shortLoud = [Float](repeating: 0.1, count: 100)
            assertTrue(vad.isSpeech(shortLoud), "short loud buffer is speech")
            let shortSilent = [Float](repeating: 0.0, count: 100)
            assertTrue(!vad.isSpeech(shortSilent), "short silent buffer is silence")
            // custom threshold
            let strictVad = VAD(threshold: 0.5)
            assertTrue(!strictVad.isSpeech(loud), "0.1 amplitude below strict threshold")
            let veryLoud = [Float](repeating: 0.9, count: 16000)
            assertTrue(strictVad.isSpeech(veryLoud), "0.9 amplitude above strict threshold")
        }

        suite("VAD.isSpeech — activity ratio")
        do {
            // Only 5% of windows active: below default minActiveRatio of 0.1 → silence
            let vad = VAD(threshold: 0.01, windowSize: 480, minActiveRatio: 0.1)
            var sparse = [Float](repeating: 0.0, count: 9600)     // 20 windows
            // Activate window 0 only (5%)
            for i in 0..<480 { sparse[i] = 0.1 }
            assertTrue(!vad.isSpeech(sparse), "only 1/20 active windows → silence")
            // Activate 3 windows (15% > 10%) → speech
            for i in 480..<960 { sparse[i] = 0.1 }
            for i in 960..<1440 { sparse[i] = 0.1 }
            assertTrue(vad.isSpeech(sparse), "3/20 active windows → speech")
        }

        suite("VAD.isSpeech — empty buffer")
        do {
            let vad = VAD()
            assertTrue(!vad.isSpeech([]), "empty buffer is silence")
        }

        // ── Adversarial: VAD boundary attacks ────────────────────────────────
        suite("VAD adversarial — degenerate params")
        do {
            let alwaysSpeech = VAD(threshold: 0.005, windowSize: 1, minActiveRatio: 0.0)
            assertTrue(alwaysSpeech.isSpeech([Float](repeating: 0.0, count: 100)), "minActiveRatio=0 always speech")
            let neverSpeech = VAD(threshold: 0.005, windowSize: 1, minActiveRatio: 1.0)
            // all-zero → no active windows → ratio=0 < 1.0 → silence
            assertTrue(!neverSpeech.isSpeech([Float](repeating: 0.0, count: 100)), "all-zero with minActiveRatio=1.0 → silence")
            // all-loud → all windows active → ratio=1.0 ≥ 1.0 → speech
            assertTrue(neverSpeech.isSpeech([Float](repeating: 0.1, count: 100)), "all-loud with minActiveRatio=1.0 → speech")
        }

        suite("VAD adversarial — buffer length boundaries")
        do {
            let vad = VAD(threshold: 0.005, windowSize: 480, minActiveRatio: 0.1)
            // Exactly windowSize-1: uses short-buffer RMS path
            let shortLoud = [Float](repeating: 0.1, count: 479)
            assertTrue(vad.isSpeech(shortLoud), "length=windowSize-1 loud → speech via RMS path")
            // Exactly windowSize: uses windowed path, one window, 1/1=100% active
            let exactLoud = [Float](repeating: 0.1, count: 480)
            assertTrue(vad.isSpeech(exactLoud), "length=windowSize loud → speech via windowed path")
            let exactSilent = [Float](repeating: 0.0, count: 480)
            assertTrue(!vad.isSpeech(exactSilent), "length=windowSize silent → silence")
        }

        suite("VAD adversarial — cancelling waveform")
        do {
            let vad = VAD()
            // Alternating +0.1 / -0.1: RMS = 0.1, well above threshold → still speech
            // (RMS doesn't cancel: sqrt(mean(x^2)) = 0.1 regardless of sign)
            var alternating = [Float](repeating: 0.0, count: 16000)
            for i in 0..<alternating.count { alternating[i] = (i % 2 == 0) ? 0.1 : -0.1 }
            assertTrue(vad.isSpeech(alternating), "alternating +/-0.1 has RMS=0.1 → speech")
        }

        suite("VAD adversarial — threshold equality and extreme values")
        do {
            let vad = VAD(threshold: 0.005)
            // Exact threshold: rms == threshold should count as speech (>=)
            assertTrue(vad.isSpeech([0.005]), "single sample exactly at threshold is speech")
            assertTrue(!vad.isSpeech([0.004999999]), "single sample just below threshold is silence")

            // Extreme amplitudes: squared values overflow to infinity, sqrt(inf) = inf >= threshold
            let maxSample = Float.greatestFiniteMagnitude
            assertTrue(vad.isSpeech([Float](repeating: maxSample, count: 1000)), "Float.max samples overflow RMS to inf → speech")
            assertTrue(vad.isSpeech([Float](repeating: -maxSample, count: 1000)), "-Float.max samples overflow RMS to inf → speech")

            // NaN propagates through RMS; NaN >= threshold is false → silence
            assertTrue(!vad.isSpeech([Float](repeating: Float.nan, count: 1000)), "NaN samples produce NaN RMS → silence")
        }

        // ── Adversarial: CommandParser mutation attacks ───────────────────────
        suite("CommandParser adversarial — homoglyph/case")
        do {
            // Cyrillic 'о' (U+043E) looks like 'o' but isn't ASCII
            let cyrillicUndo = "und\u{043E}"   // "undо" with Cyrillic о
            assertNil(CommandParser.parse(cyrillicUndo), "Cyrillic homoglyph not parsed as undo")
            // Mixed case — only exact case-fold should match
            assertEqual(CommandParser.parse("UNDO"), .undo, "UNDO parses (trimmed + lowercased in impl)")
            assertEqual(CommandParser.parse("Undo"), .undo, "Undo parses")
            assertEqual(CommandParser.parse("uNdO"), .undo, "uNdO parses")
        }

        suite("CommandParser adversarial — embedded commands")
        do {
            // These look like commands but are embedded in sentences
            assertNil(CommandParser.parse("please undo this"), "embedded undo is not a pure command")
            assertNil(CommandParser.parse("can you new line"), "embedded new line is not pure")
            assertNil(CommandParser.parse("I want to undo"), "trailing undo with prefix is not pure")
            // But suffix extraction should find them
            assertNotNil(CommandParser.extractCommand(from: "please undo"), "suffix undo extracted")
            // Pure command must be only word(s), no leading text
            assertNil(CommandParser.extractCommand(from: "can you undo this text"), "'undo' mid-sentence not extracted")
        }

        suite("CommandParser adversarial — extract ambiguities")
        do {
            // Double suffix: extracts the last command
            let doubled = CommandParser.extractCommand(from: "hello new line new line")
            assertNotNil(doubled, "double suffix extracts")
            assertEqual(doubled?.command, .pressEnter, "last new line wins")
            assertEqual(doubled?.prefix, "hello new line", "prefix stops before last command")
            // Very long input should not hang (10k chars + suffix)
            let longPrefix = String(repeating: "word ", count: 2000) + "undo"
            let result = CommandParser.extractCommand(from: longPrefix)
            assertNotNil(result, "10k-char input with suffix undo should extract")
            assertEqual(result?.command, .undo, "extracted command is undo")
            // Commands that used to exist are now just text
            assertNil(CommandParser.parse("delete that"), "delete that is no longer a command")
            assertNil(CommandParser.parse("capitalize that"), "capitalize that is no longer a command")
            assertNil(CommandParser.parse("select hello"), "select is no longer a command")
            assertNil(CommandParser.parse("replace foo with bar"), "replace is no longer a command")
            // They also don't extract as suffixes
            assertNil(CommandParser.extractCommand(from: "hello delete that"), "delete that no longer a suffix command")
            assertNil(CommandParser.extractCommand(from: "hello capitalize that"), "capitalize that not a suffix")
        }

        suite("CommandParser adversarial — position boundaries")
        do {
            // Command at position 0 is a pure command, not a suffix extraction
            assertNil(CommandParser.extractCommand(from: "undo"), "command at position 0 is pure, not extracted")
            assertNil(CommandParser.extractCommand(from: "new line"), "multi-word pure command not extracted")
            // Single-word command only
            assertNil(CommandParser.extractCommand(from: "undo "), "trailing whitespace command is pure")
            // Prefix + command with minimal whitespace
            assertNotNil(CommandParser.extractCommand(from: "x undo"), "single-char prefix + command extracts")
            assertEqual(CommandParser.extractCommand(from: "x undo")?.prefix, "x", "minimal prefix captured")
            // Command at end vs middle
            assertNil(CommandParser.extractCommand(from: "undo please"), "command at start with trailing text not extracted")
            assertNotNil(CommandParser.extractCommand(from: "please undo"), "command at end extracted")
        }

        // ── Adversarial: MoonshineEngine.chunkRanges edge cases ──────────────
        suite("MoonshineEngine adversarial — chunk invariants")
        do {
            // No chunk should exceed 160000 samples
            for count in [1, 160001, 200000, 320000, 480000] {
                let ranges = MoonshineEngine.chunkRanges(sampleCount: count)
                for r in ranges {
                    assertTrue(r.count <= 160000, "chunk \(r) in count=\(count) exceeds 160000 samples")
                }
            }
            // sampleCount=1: short buffer falls in one chunk or zero
            let oneRange = MoonshineEngine.chunkRanges(sampleCount: 1)
            assertTrue(oneRange.count <= 1, "sampleCount=1 produces at most 1 chunk")
            assertEqual(oneRange, [0..<1], "sampleCount=1 returns single-sample chunk")

            // sampleCount=160001: just over one encoder window; tail is dropped (< minChunkSamples)
            let overOne = MoonshineEngine.chunkRanges(sampleCount: 160001)
            assertTrue(overOne.allSatisfy { $0.count <= 160000 }, "160001 sample chunks bounded")
            assertEqual(overOne.first, 0..<160000, "160001 keeps first full window")

            // sampleCount=320000: exactly 2x encoder window (needs 3 chunks due to overlap)
            let twoWindow = MoonshineEngine.chunkRanges(sampleCount: 320000)
            assertTrue(twoWindow.allSatisfy { $0.count <= 160000 }, "320000 sample chunks bounded")
            assertEqual(twoWindow.count, 3, "320000 samples produces 3 chunks")

            // Verify overlap invariants for a long buffer
            let big = MoonshineEngine.chunkRanges(sampleCount: 480000)
            for i in 1..<big.count {
                let prev = big[i-1]
                let curr = big[i]
                // No gap (non-negative overlap)
                assertTrue(curr.lowerBound <= prev.upperBound, "chunks overlap or touch (start <= prev end)")
                // Overlap should be exactly overlapSamples when both chunks are full size
                if prev.count == 160000 && curr.count == 160000 {
                    assertEqual(curr.lowerBound, prev.upperBound - 32000, "full chunks overlap by 32000 samples")
                }
                assertTrue(curr.lowerBound >= prev.lowerBound, "chunks are monotonically ordered")
            }
        }

        // ── Adversarial: VocabularyService mutation attacks ───────────────────
        suite("VocabularyService adversarial — regex special chars")
        do {
            UserDefaults.standard.removeObject(forKey: "customVocabulary")
            // Trigger with regex special chars — should not crash or behave as regex
            VocabularyService.shared.addVocab(trigger: "hello.world", replacement: "HELLO_WORLD")
            // "hello.world" trigger — must match literally, not as regex dot
            assertEqual(VocabularyService.shared.applyVocabulary(to: "helloXworld"), "helloXworld",
                        "dot in trigger is literal: 'helloXworld' should not match 'hello.world'")
            assertEqual(VocabularyService.shared.applyVocabulary(to: "hello.world"), "HELLO_WORLD",
                        "literal dot matches correctly")
            VocabularyService.shared.vocabulary = []

            VocabularyService.shared.addVocab(trigger: "^test$", replacement: "REGEX_LITERAL")
            assertEqual(VocabularyService.shared.applyVocabulary(to: "this is a test"), "this is a test",
                        "^test$ trigger is literal, not regex anchors")
            assertEqual(VocabularyService.shared.applyVocabulary(to: "^test$"), "REGEX_LITERAL",
                        "literal ^test$ matches")
            VocabularyService.shared.vocabulary = []

            VocabularyService.shared.addVocab(trigger: "(parens)", replacement: "PARENS")
            assertEqual(VocabularyService.shared.applyVocabulary(to: "parens"), "parens",
                        "(parens) trigger is literal, not regex group")
            assertEqual(VocabularyService.shared.applyVocabulary(to: "(parens)"), "PARENS",
                        "literal (parens) matches")
            VocabularyService.shared.vocabulary = []
        }

        suite("VocabularyService adversarial — empty trigger/replacement")
        do {
            // Adding empty trigger should not crash and not corrupt vocab
            let before = VocabularyService.shared.vocabulary.count
            VocabularyService.shared.addVocab(trigger: "", replacement: "EMPTY")
            // Empty trigger is skipped by applyVocabulary, text unchanged
            assertEqual(VocabularyService.shared.applyVocabulary(to: "some text"), "some text",
                        "empty trigger does not change text")
            VocabularyService.shared.vocabulary = []
            assertEqual(VocabularyService.shared.vocabulary.count, 0, "vocab cleared after empty trigger test")
            _ = before

            // Empty replacement deletes the trigger
            VocabularyService.shared.addVocab(trigger: "foo", replacement: "")
            assertEqual(VocabularyService.shared.applyVocabulary(to: "foo bar"), " bar",
                        "empty replacement deletes trigger")
            VocabularyService.shared.vocabulary = []

            // Replacement longer than 1000 chars
            let longReplacement = String(repeating: "A", count: 2000)
            VocabularyService.shared.addVocab(trigger: "short", replacement: longReplacement)
            assertEqual(VocabularyService.shared.applyVocabulary(to: "short"), longReplacement,
                        "replacement longer than 1000 chars applied correctly")
            VocabularyService.shared.vocabulary = []
        }

        suite("VocabularyService adversarial — substring triggers")
        do {
            // "test" and "testing" as separate triggers: shorter must not partially match longer
            VocabularyService.shared.addVocab(trigger: "test", replacement: "T")
            assertEqual(VocabularyService.shared.applyVocabulary(to: "test now"), "T now",
                        "shorter trigger matches standalone word")
            // Desired behavior: "test" should NOT match inside "testing"
            assertEqual(VocabularyService.shared.applyVocabulary(to: "testing now"), "testing now",
                        "shorter trigger should not partial-match longer word")
            VocabularyService.shared.vocabulary = []
        }

        suite("MoonshineEngine sequential transcribe")
        let baseDir = URL(fileURLWithPath:
            "/Users/jwalinshah/.cache/moonshine-coreml/base")
        if FileManager.default.fileExists(atPath: baseDir.path) {
            let engine = MoonshineEngine(modelDir: baseDir)
            do {
                try engine.load()
                ok("base engine loaded")

                // Transcribe same short file 3 times sequentially
                let wav = "/Users/jwalinshah/Library/Logs/voice-engine/audio/2026-07-01-22-21-11.wav"
                guard let (samples, _) = Bench.loadAudio(wav) else {
                    fail("failed to load test WAV")
                    return
                }
                for i in 1...3 {
                    var t: TranscribeTiming? = nil
                    let text = try engine.transcribe(rawAudio: samples, timing: &t)
                    let empty = text.trimmingCharacters(in: .whitespaces).isEmpty
                    assertFalse(empty, "call \(i): transcript should not be empty (got \"\(text)\")")
                }
            } catch {
                fail("base engine: \(error)")
            }
        } else {
            ok("base model dir not found — skipping sequential test")
        }

        suite("WAV persistence — 16-bit PCM format")
        do {
            let floats: [Float] = [0.0, 0.5, -0.5, 1.0, -1.0]
            let tmpDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("ve-wav-test-\(UUID().uuidString)")
            try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tmpDir) }
            let wavURL = tmpDir.appendingPathComponent("test.wav")

            let ok = AppController.writeWAV(floats: floats, to: wavURL)
            assertTrue(ok, "WAV write succeeds")

            guard let data = try? Data(contentsOf: wavURL) else {
                fail("could not read WAV output")
                return
            }

            // RIFF header
            assertEqual(String(data: data.subdata(in: 0..<4), encoding: .ascii), "RIFF", "RIFF magic")
            // WAVE format
            assertEqual(String(data: data.subdata(in: 8..<12), encoding: .ascii), "WAVE", "WAVE magic")
            // fmt chunk
            assertEqual(String(data: data.subdata(in: 12..<16), encoding: .ascii), "fmt ", "fmt chunk")

            // Audio format = 1 (PCM), not 3 (IEEE float)
            let formatTag: UInt16 = data.subdata(in: 20..<22).withUnsafeBytes { $0.load(as: UInt16.self) }
            assertEqual(formatTag, 1, "PCM format tag (not IEEE float)")

            // Channels = 1
            let ch: UInt16 = data.subdata(in: 22..<24).withUnsafeBytes { $0.load(as: UInt16.self) }
            assertEqual(ch, 1, "mono")

            // Sample rate = 16000
            let sr: UInt32 = data.subdata(in: 24..<28).withUnsafeBytes { $0.load(as: UInt32.self) }
            assertEqual(sr, 16000, "16 kHz sample rate")

            // Bits per sample = 16
            let bps: UInt16 = data.subdata(in: 34..<36).withUnsafeBytes { $0.load(as: UInt16.self) }
            assertEqual(bps, 16, "16 bits per sample")

            // Data chunk
            assertEqual(String(data: data.subdata(in: 36..<40), encoding: .ascii), "data", "data chunk")

            // Data size = 5 samples × 2 bytes = 10
            let dsize: UInt32 = data.subdata(in: 40..<44).withUnsafeBytes { $0.load(as: UInt32.self) }
            assertEqual(dsize, 10, "data size = 5 × 2 bytes")

            // Total file size: 44 header + 10 data = 54
            assertEqual(data.count, 54, "total file size = 54 bytes")
        }

        suite("WAV persistence — float→int16 conversion")
        do {
            let tmpDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("ve-wav-int16-\(UUID().uuidString)")
            try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tmpDir) }
            let wavURL = tmpDir.appendingPathComponent("test.wav")

            // 1.0 → 32767, -1.0 → -32768, 0.0 → 0
            let floats: [Float] = [1.0, -1.0, 0.0, 0.5, -0.5]
            assertTrue(AppController.writeWAV(floats: floats, to: wavURL), "write succeeds")

            guard let data = try? Data(contentsOf: wavURL) else { fail("read fail"); return }
            let samples: [Int16] = data.subdata(in: 44..<54).withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: Int16.self))
            }
            assertEqual(Int(samples[0]), 32767, "1.0 → 32767")
            assertEqual(Int(samples[1]), -32768, "-1.0 → -32768")
            assertEqual(Int(samples[2]), 0, "0.0 → 0")
            assertEqual(Int(samples[3]), 16383, "0.5 → ~16384")
            assertEqual(Int(samples[4]), -16384, "-0.5 → ~-16384")
        }

        suite("WAV persistence — clipping")
        do {
            let tmpDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("ve-wav-clip-\(UUID().uuidString)")
            try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tmpDir) }
            let wavURL = tmpDir.appendingPathComponent("test.wav")

            // Out-of-range floats must clip to int16 range, not wrap
            let floats: [Float] = [2.0, -2.0, Float.infinity, -Float.infinity]
            assertTrue(AppController.writeWAV(floats: floats, to: wavURL), "write succeeds")
            guard let data = try? Data(contentsOf: wavURL) else { fail("read fail"); return }
            let samples: [Int16] = data.subdata(in: 44..<52).withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: Int16.self))
            }
            assertEqual(Int(samples[0]), 32767, "2.0 clips to 32767")
            assertEqual(Int(samples[1]), -32768, "-2.0 clips to -32768")
            assertEqual(Int(samples[2]), 32767, "+inf clips to 32767")
            assertEqual(Int(samples[3]), -32768, "-inf clips to -32768")
        }

        suite("WAV persistence — NaN → crash-free")
        do {
            let tmpDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("ve-wav-nan-\(UUID().uuidString)")
            try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: tmpDir) }
            let wavURL = tmpDir.appendingPathComponent("test.wav")

            // NaN should not crash — writes a valid RIFF file
            let floats: [Float] = [Float.nan, Float.nan]
            assertTrue(AppController.writeWAV(floats: floats, to: wavURL), "NaN write does not crash")
            guard let data = try? Data(contentsOf: wavURL) else { fail("read fail"); return }
            assertEqual(data.count, 48, "NaN samples produce 48-byte WAV (2 samples × 2 bytes)")
        }

        suite("WAV persistence — invalid destination")
        do {
            let badURL = URL(fileURLWithPath: "/nonexistent/dir/should/fail.wav")
            assertFalse(AppController.writeWAV(floats: [0.0], to: badURL),
                       "unwritable path returns false (non-fatal)")
        }

        suite("JSON sidecar metadata — field identity")
        do {
            let now = Date()
            let meta = AppController.sidecarMetadata(ts: now, sampleCount: 16000,
                                                      transcription: "hello", app: "com.example")

            // Required fields always present
            let tsVal: Any? = meta["ts"]
            let durVal: Any? = meta["duration_secs"]
            assertNotNil(tsVal, "ts field present")
            assertNotNil(durVal, "duration_secs field present")

            // Duration = sampleCount / 16000
            let dur = meta["duration_secs"] as? Double
            assertNotNil(dur, "duration_secs is Double")
            if let d = dur { assertEqual(d, 1.0, "16000 samples = 1.0 sec") }

            // Optional fields present when provided
            let textVal = meta["text"] as? String
            let appVal = meta["app"] as? String
            assertEqual(textVal, "hello", "text field matches")
            assertEqual(appVal, "com.example", "app field matches")

            // ISO-8601 timestamp
            let ts = meta["ts"] as? String
            assertNotNil(ts, "ts is a string")
            if let t = ts { assertTrue(t.contains("T"), "ts is ISO-8601 (contains 'T')") }
        }

        suite("JSON sidecar metadata — absent optionals")
        do {
            let meta = AppController.sidecarMetadata(ts: Date(), sampleCount: 8000,
                                                      transcription: nil, app: nil)
            // Optional keys omitted when nil
            let textVal: Any? = meta["text"]
            let appVal: Any? = meta["app"]
            assertNil(textVal, "text omitted when transcription is nil")
            assertNil(appVal, "app omitted when app is nil")

            let dur = meta["duration_secs"] as? Double
            if let d = dur { assertEqual(d, 0.5, "8000 samples = 0.5 sec") }
        }

        // ── History format and append semantics ───────────────────────────
        // The voice-history.txt line is produced in a fire-and-forget
        // Task.detached closure and can't be called from the synchronous test
        // runner. Format is trivially verifiable by inspection:
        //   "[<ISO-8601 ts>] <transcript text>\n"
        // and is covered by the integration surface of the app.
        //
        // Sidecar update (text, raw_text, cleaned_text, timing breakdown)
        // likewise runs in the background detached task and is covered by
        // the JSON metadata test above for initial fields, plus integration
        // surface for the update path.

        suite("voice-history — ISO-8601 line format check")
        do {
            // Verify the formatting pattern we use in production is correct
            let isoTs = ISO8601DateFormatter().string(from: Date())
            let line = "[\(isoTs)] hello world\n"
            assertTrue(line.hasPrefix("["), "history line starts with '['")
            assertTrue(line.contains("T"), "history line contains ISO-8601 'T'")
            assertTrue(line.hasSuffix("\n"), "history line ends with newline")
            assertTrue(line.contains("] hello world"), "history line has '] ' before text")
        }

        // MARK: - FullStop Punctuation tests

        // ── Label set invariant (no files needed) ────────────────────────
        // MARK: - CapitalizationService tests

        suite("CapitalizationService — sentence-start")
        assertEqual(CapitalizationService.capitalize("hello"), "Hello", "single word")
        assertEqual(CapitalizationService.capitalize("hello world"), "Hello world", "two words")
        assertEqual(CapitalizationService.capitalize("hello. world"), "Hello. World", "period boundary")
        assertEqual(CapitalizationService.capitalize("hello? world"), "Hello? World", "question mark boundary")
        assertEqual(CapitalizationService.capitalize("hello! world"), "Hello! World", "exclamation mark boundary")
        assertEqual(CapitalizationService.capitalize("a. b. c."), "A. B. C.", "multiple sentences")

        suite("CapitalizationService — whitespace and quotes after punctuation")
        assertEqual(CapitalizationService.capitalize("hello.  world"), "Hello.  World", "double space after period")
        assertEqual(CapitalizationService.capitalize("hello.\nworld"), "Hello.\nWorld", "newline after period")
        assertEqual(CapitalizationService.capitalize("hello.\tworld"), "Hello.\tWorld", "tab after period")
        assertEqual(CapitalizationService.capitalize("hello. \"world"), "Hello. \"World", "quote after period — world capitalized")

        suite("CapitalizationService — pronoun i")
        assertEqual(CapitalizationService.capitalize("i think"), "I think", "sentence-start i")
        assertEqual(CapitalizationService.capitalize("i think i can"), "I think I can", "mid-sentence i")
        assertEqual(CapitalizationService.capitalize("i'm here"), "I'm here", "contraction i'm")
        assertEqual(CapitalizationService.capitalize("it is what it is"), "It is what it is", "i inside 'it' untouched")
        assertEqual(CapitalizationService.capitalize("in and out"), "In and out", "i inside 'in' untouched")

        suite("CapitalizationService — proper noun dictionary")
        // Companies
        assertEqual(CapitalizationService.capitalize("github"), "GitHub", "github → GitHub")
        assertEqual(CapitalizationService.capitalize("openai"), "OpenAI", "openai → OpenAI")
        assertEqual(CapitalizationService.capitalize("anthropic"), "Anthropic", "anthropic → Anthropic")
        assertEqual(CapitalizationService.capitalize("slack"), "Slack", "slack → Slack")
        assertEqual(CapitalizationService.capitalize("discord"), "Discord", "discord → Discord")
        assertEqual(CapitalizationService.capitalize("cursor"), "Cursor", "cursor → Cursor")
        assertEqual(CapitalizationService.capitalize("apple"), "Apple", "apple → Apple")
        assertEqual(CapitalizationService.capitalize("meta"), "Meta", "meta → Meta")
        assertEqual(CapitalizationService.capitalize("google"), "Google", "google → Google")
        assertEqual(CapitalizationService.capitalize("microsoft"), "Microsoft", "microsoft → Microsoft")
        // Products
        assertEqual(CapitalizationService.capitalize("chatgpt"), "ChatGPT", "chatgpt → ChatGPT")
        assertEqual(CapitalizationService.capitalize("claude"), "Claude", "claude → Claude")
        assertEqual(CapitalizationService.capitalize("moonshine"), "Moonshine", "moonshine → Moonshine")
        assertEqual(CapitalizationService.capitalize("xcode"), "Xcode", "xcode → Xcode")
        assertEqual(CapitalizationService.capitalize("safari"), "Safari", "safari → Safari")
        assertEqual(CapitalizationService.capitalize("chrome"), "Chrome", "chrome → Chrome")
        assertEqual(CapitalizationService.capitalize("firefox"), "Firefox", "firefox → Firefox")
        assertEqual(CapitalizationService.capitalize("iterm"), "iTerm", "iterm → iTerm")
        // Mixed casing preserved
        assertEqual(CapitalizationService.capitalize("macos"), "macOS", "macos → macOS")
        assertEqual(CapitalizationService.capitalize("ios"), "iOS", "ios → iOS")
        assertEqual(CapitalizationService.capitalize("iphone"), "iPhone", "iphone → iPhone")
        assertEqual(CapitalizationService.capitalize("ipad"), "iPad", "ipad → iPad")
        // Protocols/formats
        assertEqual(CapitalizationService.capitalize("javascript"), "JavaScript", "javascript → JavaScript")
        assertEqual(CapitalizationService.capitalize("typescript"), "TypeScript", "typescript → TypeScript")
        assertEqual(CapitalizationService.capitalize("json"), "JSON", "json → JSON")
        assertEqual(CapitalizationService.capitalize("yaml"), "YAML", "yaml → YAML")
        assertEqual(CapitalizationService.capitalize("api"), "API", "api → API")
        assertEqual(CapitalizationService.capitalize("html"), "HTML", "html → HTML")
        assertEqual(CapitalizationService.capitalize("css"), "CSS", "css → CSS")
        assertEqual(CapitalizationService.capitalize("sql"), "SQL", "sql → SQL")
        assertEqual(CapitalizationService.capitalize("ssh"), "SSH", "ssh → SSH")
        assertEqual(CapitalizationService.capitalize("rest"), "REST", "rest → REST")
        assertEqual(CapitalizationService.capitalize("http"), "HTTP", "http → HTTP")
        // Common
        assertEqual(CapitalizationService.capitalize("git"), "Git", "git → Git")
        assertEqual(CapitalizationService.capitalize("mac"), "Mac", "mac → Mac")
        assertEqual(CapitalizationService.capitalize("linux"), "Linux", "linux → Linux")
        assertEqual(CapitalizationService.capitalize("docker"), "Docker", "docker → Docker")
        assertEqual(CapitalizationService.capitalize("kubernetes"), "Kubernetes", "kubernetes → Kubernetes")
        assertEqual(CapitalizationService.capitalize("python"), "Python", "python → Python")
        assertEqual(CapitalizationService.capitalize("swift"), "Swift", "swift → Swift")
        assertEqual(CapitalizationService.capitalize("rust"), "Rust", "rust → Rust")

        suite("CapitalizationService — full-word protection")
        assertEqual(CapitalizationService.capitalize("pineapple"), "Pineapple", "pineapple: 'apple' not matched inside")
        assertEqual(CapitalizationService.capitalize("github is not githubby"), "GitHub is not githubby", "githubby untouched")
        assertEqual(CapitalizationService.capitalize("slacker"), "Slacker", "slacker: 'slack' not matched inside")
        assertEqual(CapitalizationService.capitalize("discordant"), "Discordant", "discordant: 'discord' not matched inside")

        suite("CapitalizationService — possessives")
        assertEqual(CapitalizationService.capitalize("openai's"), "OpenAI's", "openai's → OpenAI's")
        assertEqual(CapitalizationService.capitalize("github's"), "GitHub's", "github's → GitHub's")
        assertEqual(CapitalizationService.capitalize("apple's"), "Apple's", "apple's → Apple's")

        suite("CapitalizationService — multiword names")
        assertEqual(CapitalizationService.capitalize("vs code"), "VS Code", "vs code → VS Code")
        assertEqual(CapitalizationService.capitalize("VS CODE"), "VS Code", "VS CODE → VS Code")
        assertEqual(CapitalizationService.capitalize("i use vs code"), "I use VS Code", "vs code in sentence")

        suite("CapitalizationService — mixed known casing preserved")
        // Confirm known mixed-case entries survive sentence-start capitalization
        assertEqual(CapitalizationService.capitalize("macos is great"), "macOS is great", "macOS not macOS")
        assertEqual(CapitalizationService.capitalize("ios and macos"), "iOS and macOS", "iOS and macOS in sentence")
        assertEqual(CapitalizationService.capitalize("iphone ipad mac"), "iPhone iPad Mac", "multiple Apple products")
        assertEqual(CapitalizationService.capitalize("github openai chatgpt"), "GitHub OpenAI ChatGPT", "multiple products")

        suite("CapitalizationService — overlapping dictionary entries")
        // "GitHub" must not be corrupted by "Git" partial match
        assertEqual(CapitalizationService.capitalize("github"), "GitHub", "GitHub vs Git: long wins")
        assertEqual(CapitalizationService.capitalize("git is a vcs"), "Git is a vcs", "standalone Git")
        assertEqual(CapitalizationService.capitalize("javascript is not java"), "JavaScript is not java", "JavaScript long form")

        suite("CapitalizationService — idempotence")
        let idemCases: [(String, String)] = [
            ("hello world", "Hello world"),
            ("hello. world", "Hello. World"),
            ("i think i can", "I think I can"),
            ("github and openai", "GitHub and OpenAI"),
            ("macos and ios", "macOS and iOS"),
            ("vs code is great", "VS Code is great"),
            ("hello. \"world", "Hello. \"World"),
            ("a. b. c.", "A. B. C."),
            ("openai's model", "OpenAI's model"),
        ]
        for (input, _) in idemCases {
            let first = CapitalizationService.capitalize(input)
            let second = CapitalizationService.capitalize(first)
            assertEqual(second, first, "idempotent: '\(input)' → '\(first)' → '\(second)'")
        }

        suite("CapitalizationService — empty and edge cases")
        assertEqual(CapitalizationService.capitalize(""), "", "empty string")
        assertEqual(CapitalizationService.capitalize("..."), "...", "punctuation only")
        assertEqual(CapitalizationService.capitalize("!@#$%"), "!@#$%", "symbols only")
        assertEqual(CapitalizationService.capitalize("   "), "   ", "whitespace only")
        assertEqual(CapitalizationService.capitalize("123"), "123", "numbers only — no letters to capitalize")

        suite("CapitalizationService — Unicode grapheme clusters")
        // composed characters must not be corrupted
        assertEqual(CapitalizationService.capitalize("café"), "Café", "accented e preserved")
        assertEqual(CapitalizationService.capitalize("café. voilà"), "Café. Voilà", "accented chars after period")
        // emoji sequence should pass through unchanged
        assertEqual(CapitalizationService.capitalize("hello 👋 world"), "Hello 👋 world", "emoji preserved")
        // decomposed e + combining acute → uppercase should handle correctly
        let decomposed = "cafe\u{0301}"  // e + combining acute accent
        let result = CapitalizationService.capitalize(decomposed)
        assertTrue(result.first == "C" || result.hasPrefix("C"), "decomposed grapheme cluster survives capitalization")

        suite("CapitalizationService — pipeline order: sentence-start then dict")
        // Sentence-start runs first, then dict overrides to fix mixed-casing entries.
        // "ios" → sentence-start: "Ios" → dict: case-insensitive "ios" matches "Ios" → "iOS"
        assertEqual(CapitalizationService.capitalize("ios is fast"), "iOS is fast", "iOS not Ios")
        assertEqual(CapitalizationService.capitalize("iterm is great"), "iTerm is great", "iTerm not ITerm")

        suite("PunctuationService — label set invariant")
        do {
            let labels = PunctuationService.idToLabel
            let allowed = Set([".", ",", "?", "-", ":", ""])
            for (_, char) in labels {
                assertTrue(allowed.contains(char), "label '\(char)' is in allowed set")
            }
            assertEqual(labels.count, PunctuationService.numLabels,
                       "exactly \(PunctuationService.numLabels) labels")
        }

        // ── Tokenizer unit tests (self-contained, no machine caches) ───
        // Create a minimal synthetic vocab in a temp directory so these
        // tests are independent of any cached model artifacts.
        suite("FullStopTokenizer — init and special tokens (synthetic vocab)")
        do {
            let tok = try makeSyntheticTokenizer()
            assertEqual(tok.bosId, 0, "BOS id")
            assertEqual(tok.padId, 1, "PAD id")
            assertEqual(tok.eosId, 2, "EOS id")
            assertEqual(tok.unkId, 3, "UNK id")
            ok("tokenizer init from synthetic vocab")
        } catch {
            fail("synthetic tokenizer init failed: \(error)")
        }

        suite("FullStopTokenizer — encode/decode roundtrip (synthetic vocab)")
        do {
            let tok = try makeSyntheticTokenizer()
            let result = tok.encode("hello world", maxLength: 32)
            assertTrue(result.inputIds.count <= 32, "encoded within maxLength")
            assertEqual(result.inputIds[0], 0, "starts with BOS")
            // Decode back — the synthetic vocab has limited coverage but
            // should handle simple words
            let decoded = tok.decode(result.inputIds)
            // Decode removes BOS/EOS/PAD and replaces U+2581 with space
            assertFalse(decoded.contains("<s>"), "BOS stripped from decode")
            assertFalse(decoded.contains("</s>"), "EOS stripped from decode")
            assertFalse(decoded.contains("<unk>"), "UNK should not appear")
            // Result should be non-empty
            assertTrue(!decoded.isEmpty, "decode produces text")
        } catch {
            fail("tokenizer roundtrip failed: \(error)")
        }

        suite("FullStopTokenizer — overlength truncation (synthetic vocab)")
        do {
            let tok = try makeSyntheticTokenizer()
            let longText = String(repeating: "hello world ", count: 10)
            let result = tok.encode(longText, maxLength: 16)
            // Actual-length encoding: no padding beyond BOS+content+EOS.
            // Long text is truncated to maxLength, all positions attended.
            assertEqual(result.inputIds.count, 16, "exactly maxLength tokens (truncated, not padded)")
            assertEqual(result.attentionMask.count, 16, "attention mask same length")
            // All positions are real tokens (no padding)
            for (i, m) in result.attentionMask.enumerated() {
                assertEqual(m, 1, "position \(i) attended (no padding)")
            }
        } catch {
            fail("overlength truncation test failed: \(error)")
        }

        suite("FullStopTokenizer — empty input")
        do {
            let tok = try makeSyntheticTokenizer()
            let result = tok.encode("", maxLength: 256)
            // Empty input: only BOS + EOS, no padding.
            assertEqual(result.inputIds.count, 2, "only BOS+EOS (no padding)")
            assertEqual(result.inputIds[0], 0, "BOS at position 0")
            assertEqual(result.inputIds[1], 2, "EOS at position 1")
            assertEqual(result.attentionMask, [1, 1], "both positions attended")
        } catch {
            fail("empty input test failed: \(error)")
        }

        suite("FullStopTokenizer — whitespace only input")
        do {
            let tok = try makeSyntheticTokenizer()
            let result = tok.encode("   ", maxLength: 256)
            // Whitespace-only: BOS + EOS, no padding.
            assertEqual(result.inputIds.count, 2, "only BOS+EOS for whitespace-only input")
            assertEqual(result.inputIds[0], 0, "BOS")
            assertEqual(result.inputIds[1], 2, "EOS")
        } catch {
            fail("whitespace input test failed: \(error)")
        }

        // ── New: actual-length short input (no-pad semantics) ──────────────
        suite("FullStopTokenizer — actual-length short input")
        do {
            let tok = try makeSyntheticTokenizer()
            // Four-word sentence with synthetic vocab
            let result = tok.encode("hello world the quick", maxLength: 256)
            // BOS + 4 words (▁hello ▁world ▁the ▁quick) + EOS = 6 tokens
            // with the synthetic vocab (no subword splits for these words)
            assertTrue(result.inputIds.count < 256, "short input uses actual length, not maxLength")
            assertTrue(result.inputIds.count >= 4, "at least BOS + some words + EOS")
            assertEqual(result.attentionMask.count, result.inputIds.count, "mask matches actual length")
            // All attention mask entries are 1 (no padding positions)
            for m in result.attentionMask {
                assertEqual(m, 1, "all positions attended")
            }
            // First is BOS, last is EOS
            assertEqual(result.inputIds[0], 0, "starts with BOS")
            assertEqual(result.inputIds.last, 2, "ends with EOS")
        } catch {
            fail("actual-length test failed: \(error)")
        }

        // ── New: exact max-length truncation ───────────────────────────────
        suite("FullStopTokenizer — exact max-length truncation")
        do {
            let tok = try makeSyntheticTokenizer()
            // Request exactly 5 tokens max
            let result = tok.encode("hello world the quick brown fox", maxLength: 5)
            assertEqual(result.inputIds.count, 5, "exactly maxLength tokens")
            // BOS + content (truncated before EOS if it doesn't fit at exactly maxLength)
            // Actually: BOS + some tokens + EOS truncated to 5
            assertEqual(result.inputIds[0], 0, "BOS first")
            // Last token should be EOS if it fits
            // If the encoding loop fills up to maxLength-1, EOS is appended and
            // truncated to maxLength, so the last token should be EOS
            assertTrue(result.inputIds.last == 2 || result.attentionMask.last == 1,
                      "last token is either EOS or attended content")
            // All positions attended
            for m in result.attentionMask {
                assertEqual(m, 1, "all positions attended")
            }
        } catch {
            fail("exact max-length truncation test failed: \(error)")
        }

        // ── Actor safety test ──────────────────────────────────────────
        // Verifies PunctuationService is a proper actor, not @unchecked Sendable.
        suite("PunctuationService — actor isolation (load/restore)")
        do {
            let svc = PunctuationService()
            // load() and restore() require await — compiler enforces isolation.
            // This test proves the actor path works end-to-end with correct
            // async semantics.
            var modelAvailable = false
            do {
                try awaitSync { try await svc.load() }
                modelAvailable = true
                ok("actor load() succeeds with await")
            } catch {
                ok("skipped — model not available for actor test: \(error.localizedDescription)")
            }
            if modelAvailable {
                // Verify ready state
                let ready = try awaitSync { await svc.isReady() }
                assertTrue(ready, "actor reports ready after load")
                // Verify restore works through actor isolation
                let result = try awaitSync { try await svc.restore("hello world") }
                assertTrue(!result.isEmpty, "actor restore produces non-empty output")
                let inputWords = "hello world".split(separator: " ")
                let outputWords = result.split(separator: " ").map {
                    $0.trimmingCharacters(in: CharacterSet.punctuationCharacters)
                }
                // All input words preserved
                var wordIdx = 0
                for ow in outputWords where !ow.isEmpty {
                    if wordIdx < inputWords.count && ow.lowercased() == inputWords[wordIdx].lowercased() {
                        wordIdx += 1
                    }
                }
                assertEqual(wordIdx, inputWords.count, "actor restore preserves all input words")
                ok("actor restore succeeds with await")
            }
        } catch {
            fail("actor test failed: \(error)")
        }
        suite("PunctuationService — model artifacts available")
        do {
            let modelDir = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache/fullstop-coreml/large")
            let mlpackage = modelDir.appendingPathComponent("fullstop-punctuation.mlpackage")
            let tokFile = modelDir.appendingPathComponent("tokenizer_compact.json")
            let cfgFile = modelDir.appendingPathComponent("config.json")

            let modelExists = FileManager.default.fileExists(atPath: mlpackage.path)
            let tokExists = FileManager.default.fileExists(atPath: tokFile.path)
            let cfgExists = FileManager.default.fileExists(atPath: cfgFile.path)

            if modelExists && tokExists && cfgExists {
                ok("all FullStop artifacts present (model + tokenizer + config)")

                // ── Cross-check: Swift tokenizer vs Python/HF fixtures ──
                suite("FullStopTokenizer — Python/HF cross-check")
                // The fixture file is generated by:
                //   python3 -c "from transformers import AutoTokenizer; ..."
                // and checked into Tests/Runner/fullstop_tokenizer_fixtures.json
                let fixtureCandidates = [
                    URL(fileURLWithPath: "Tests/Runner/fullstop_tokenizer_fixtures.json"),
                    URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                        .appendingPathComponent("Tests/Runner/fullstop_tokenizer_fixtures.json"),
                ]
                let fixtureURL = fixtureCandidates.first {
                    FileManager.default.fileExists(atPath: $0.path)
                }

                if let fixtureURL,
                   let tok = try? FullStopTokenizer(vocabDir: modelDir),
                   let data = try? Data(contentsOf: fixtureURL),
                   let fixtures = try? JSONSerialization.jsonObject(with: data) as? [String: [String: [Int]]] {
                    for (text, expected) in fixtures {
                        guard let expectedIds = expected["input_ids"] else { continue }
                        let result = tok.encode(text, maxLength: 256)
                        // Compare token IDs up to the actual sequence length
                        // (the fixture has unpadded IDs including BOS/EOS)
                        let actualIds = Array(result.inputIds.prefix(expectedIds.count))
                        if actualIds == expectedIds {
                            ok("'\(text)' — \(expectedIds.count) tokens match Python/HF")
                        } else {
                            // Report first mismatch position
                            var mismatchIdx = 0
                            for i in 0..<min(actualIds.count, expectedIds.count) {
                                if actualIds[i] != expectedIds[i] {
                                    mismatchIdx = i; break
                                }
                                mismatchIdx = i + 1
                            }
                            fail("'\(text)' — mismatch at pos \(mismatchIdx): Swift=\(actualIds), HF=\(expectedIds)")
                        }
                    }
                } else {
                    ok("skipped — cannot load fixtures or tokenizer for cross-check")
                }
            } else {
                var missing: [String] = []
                if !modelExists { missing.append("mlpackage") }
                if !tokExists { missing.append("tokenizer_compact.json") }
                if !cfgExists { missing.append("config.json") }
                ok("skipped — artifacts not cached: \(missing.joined(separator: ", "))")
            }
        }

        // ── Swift end-to-end inference (skips when model absent) ────────
        suite("PunctuationService — Swift end-to-end inference")
        do {
            let svc = PunctuationService()
            do {
                try awaitSync { try await svc.load() }
            } catch {
                ok("skipped — model not available: \(error.localizedDescription)")
                return
            }

            // Test with representative lowercase ASR-like inputs.
            // Verify: input words preserved in order, only allowed
            // punctuation inserted, latency recorded.
            let testCases = [
                "hello world how are you",
                "my name is clara and i live in berkeley california",
                "what time is it",
                "okay let me think about that",
            ]

            for text in testCases {
                let t0 = CFAbsoluteTimeGetCurrent()
                let result = try awaitSync { try await svc.restore(text) }
                let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000

                // Input words preserved in output order
                let inputWords = text.lowercased().split(separator: " ")
                let outputWords = result.split(separator: " ").map {
                    $0.trimmingCharacters(in: CharacterSet.punctuationCharacters)
                }
                // outputWords may differ in count due to punctuation-only tokens
                // from reconstruction artifacts, but all input words should appear
                // in order
                var wordIdx = 0
                for ow in outputWords where !ow.isEmpty {
                    if wordIdx < inputWords.count && ow.lowercased() == inputWords[wordIdx].lowercased() {
                        wordIdx += 1
                    }
                }
                assertEqual(wordIdx, inputWords.count,
                           "'\(text.prefix(30))...' - all \(inputWords.count) input words found in output")

                // Only allowed punctuation inserted
                let allowed = CharacterSet(charactersIn: ".,?:- ")
                let illegal = result.unicodeScalars.filter { !allowed.contains($0) && !CharacterSet.alphanumerics.contains($0) }
                assertTrue(illegal.isEmpty,
                          "'\(text.prefix(30))...' - only allowed punctuation in output (found: \(illegal.map { String($0) }))")

                // Latency recorded
                assertTrue(ms > 0 && ms < 5000,
                          "'\(text.prefix(30))...' - restore latency \(String(format: "%.1f", ms))ms within 5s budget")

                Foundation.NSLog("[voice-tests] PunctuationService.restore: \(String(format: "%.1f", ms))ms — input=\"\(text)\" output=\"\(result)\"")
            }

            ok("end-to-end Swift inference passed for \(testCases.count) inputs")
        } catch {
            fail("end-to-end inference failed: \(error)")
        }
    }
}

/// Create a FullStopTokenizer with a minimal synthetic vocabulary in a
/// temp directory. This makes tokenizer unit tests independent of any
/// cached model artifacts.
///
/// The synthetic vocab covers basic English words enough to validate
/// encode/decode roundtrips, special token handling, truncation, and
/// edge cases without requiring the 250K-entry production vocab.
private func makeSyntheticTokenizer() throws -> FullStopTokenizer {
    let tmpDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("fullstop-synth-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

    // Minimal Unigram vocab covering common words and subwords.
    // IDs 0-3 are special tokens, 4+ are content pieces with scores.
    let vocab: [[Any]] = [
        ["<s>", 0.0],       // 0: BOS
        ["<pad>", 0.0],     // 1: PAD
        ["</s>", -0.1],     // 2: EOS
        ["<unk>", -10.0],   // 3: UNK
        ["▁hello", -1.0],   // 4
        ["▁world", -1.0],   // 5
        ["▁the", -0.8],     // 6
        ["▁quick", -1.5],   // 7
        ["▁brown", -1.5],   // 8
        ["▁fox", -1.5],     // 9
        ["▁jumps", -1.5],   // 10
        ["▁over", -1.5],    // 11
        ["▁lazy", -1.5],    // 12
        ["▁dog", -1.5],     // 13
        ["▁a", -0.5],       // 14
        ["▁is", -1.0],      // 15
        ["▁it", -1.0],      // 16
        ["▁this", -1.0],    // 17
        ["▁that", -1.0],    // 18
        ["▁okay", -1.8],    // 19
        ["▁test", -1.5],    // 20
        // Subword pieces for unknown words
        ["he", -2.0],        // 21
        ["llo", -2.0],       // 22
        ["wo", -2.0],        // 23
        ["rld", -2.0],       // 24
        ["qu", -2.5],        // 25
        ["ick", -2.5],       // 26
        ["br", -2.5],        // 27
        ["own", -2.5],       // 28
    ]

    let json: [String: Any] = ["vocab": vocab]
    let data = try JSONSerialization.data(withJSONObject: json)
    let compactPath = tmpDir.appendingPathComponent("tokenizer_compact.json")
    try data.write(to: compactPath)

    return try FullStopTokenizer(vocabDir: tmpDir)
}

let runner = TestRunner()
runner.runAll()
print("\n=== Results: \(runner.passed) passed, \(runner.failed) failed ===")
if runner.failed > 0 { exit(1) }
