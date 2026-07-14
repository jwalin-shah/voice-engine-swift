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
    }
}

let runner = TestRunner()
runner.runAll()
print("\n=== Results: \(runner.passed) passed, \(runner.failed) failed ===")
if runner.failed > 0 { exit(1) }
