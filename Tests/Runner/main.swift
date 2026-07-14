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
        assertEqual(CommandParser.parse("new line"), .newLine, "new line")
        assertEqual(CommandParser.parse("newline"), .newLine, "newline")
        assertEqual(CommandParser.parse("new paragraph"), .newParagraph, "new paragraph")
        assertEqual(CommandParser.parse("tab"), .tab, "tab")
        assertEqual(CommandParser.parse("delete that"), .deleteThat, "delete that")
        assertEqual(CommandParser.parse("delete this"), .deleteThat, "delete this")
        assertEqual(CommandParser.parse("capitalize that"), .capitalizeThat, "capitalize that")
        assertEqual(CommandParser.parse("uppercase that"), .capitalizeThat, "uppercase that")
        assertEqual(CommandParser.parse("lowercase that"), .lowercaseThat, "lowercase that")
        assertEqual(CommandParser.parse("select hello world"), .select("hello world"), "select hello world")
        assertEqual(CommandParser.parse("SELECT HELLO"), .select("HELLO"), "SELECT HELLO parsed")
        assertEqual(CommandParser.parse("replace foo with bar"), .replace("foo", "bar"), "replace foo with bar")
        assertEqual(CommandParser.parse("replace a with b c"), .replace("a", "b c"), "replace a with b c")
        assertNil(CommandParser.parse("select "), "select no target")
        assertNil(CommandParser.parse("select"), "select exact")
        assertNil(CommandParser.parse("replace foo bar"), "replace without with")
        assertNil(CommandParser.parse("replace  with "), "replace empty args")
        assertNil(CommandParser.parse("hello world"), "no command")
        assertNil(CommandParser.parse(""), "empty string")
        assertNil(CommandParser.parse("   "), "whitespace only")
        assertEqual(CommandParser.parse("  undo  "), .undo, "padded undo")
        suite("CommandParser.extractCommand")
        do { let r = CommandParser.extractCommand(from: "hello world undo"); assertEqual(r?.prefix, "hello world"); assertEqual(r?.command, .undo) }
        assertEqual(CommandParser.extractCommand(from: "hello new line")?.command, .newLine)
        assertEqual(CommandParser.extractCommand(from: "hello world capitalize that")?.command, .capitalizeThat)
        do { let r = CommandParser.extractCommand(from: "hello world select foo"); assertEqual(r?.prefix, "hello world"); assertEqual(r?.command, .select("foo")) }
        assertEqual(CommandParser.extractCommand(from: "hello world replace old with new")?.command, .replace("old", "new"))
        assertNil(CommandParser.extractCommand(from: "undo"), "pure undo skipped")
        assertNil(CommandParser.extractCommand(from: "new line"), "pure new line skipped")
        assertNil(CommandParser.extractCommand(from: "helloundo"), "no word boundary")
        assertNil(CommandParser.extractCommand(from: "hello_undo"), "underscore boundary")
        assertNil(CommandParser.extractCommand(from: ""), "empty string")
        assertEqual(CommandParser.extractCommand(from: "hello world delete that")?.command, .deleteThat)
        assertEqual(CommandParser.extractCommand(from: "hello lowercase that")?.command, .lowercaseThat)
        assertEqual(CommandParser.extractCommand(from: "hello uppercase that")?.command, .capitalizeThat)
        assertEqual(CommandParser.extractCommand(from: "hello tab")?.command, .tab)
        assertEqual(CommandParser.extractCommand(from: "hello new paragraph")?.command, .newParagraph)
        suite("CommandParser.Equatable")
        assertEqual(CommandParser.VoiceCommand.select("a"), .select("a"))
        assertEqual(CommandParser.VoiceCommand.undo, .undo)
        assertTrue(CommandParser.VoiceCommand.select("a") != .select("b"))
        assertTrue(CommandParser.VoiceCommand.undo != .newLine)
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
            // "select" with no target as suffix
            assertNil(CommandParser.extractCommand(from: "hello select"), "select with no target → nil")
            // "replace" with no valid old/new pair
            assertNil(CommandParser.extractCommand(from: "hello replace"), "replace with no args → nil")
            assertNil(CommandParser.extractCommand(from: "hello replace foo"), "replace without with → nil")
            // Double suffix: extracts the last command
            let doubled = CommandParser.extractCommand(from: "hello delete that delete that")
            assertNotNil(doubled, "double suffix extracts")
            assertEqual(doubled?.command, .deleteThat, "last delete that wins")
            assertEqual(doubled?.prefix, "hello delete that", "prefix stops before last command")
            // Very long input should not hang (10k chars + suffix)
            let longPrefix = String(repeating: "word ", count: 2000) + "undo"
            let result = CommandParser.extractCommand(from: longPrefix)
            assertNotNil(result, "10k-char input with suffix undo should extract")
            assertEqual(result?.command, .undo, "extracted command is undo")
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

        // MARK: - Punctuation Service tests

        suite("PunctuationService — tokenizer init")
        do {
            let modelDir = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache/fullstop-coreml/large")
            if FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("tokenizer_compact.json").path),
               let tok = try? FullStopTokenizer(vocabDir: modelDir) {
                assertEqual(tok.bosId, 0, "BOS id")
                assertEqual(tok.eosId, 2, "EOS id")
                assertEqual(tok.padId, 1, "PAD id")
                assertEqual(tok.unkId, 3, "UNK id")
                ok("tokenizer loaded")
            } else {
                fail("tokenizer_compact.json not found")
            }
        }

        suite("PunctuationService — tokenizer encode/decode")
        do {
            let modelDir = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache/fullstop-coreml/large")
            guard FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("tokenizer_compact.json").path) else {
                fail("tokenizer file missing")
                return
            }
            guard let tok = try? FullStopTokenizer(vocabDir: modelDir) else { fail("tokenizer init failed"); return }

            let text = "hello world"
            let result = tok.encode(text, maxLength: 256)
            assertTrue(result.inputIds.count <= 256, "encoded within maxLength")
            assertEqual(result.inputIds[0], 0, "starts with BOS")

            let decoded = tok.decode(result.inputIds)
            assertEqual(decoded, text, "encode/decode roundtrip")
        }

        suite("PunctuationService — label set invariant")
        do {
            let labels = [0: "", 1: ".", 2: ",", 3: "?", 4: "-", 5: ":"]
            let allowed = Set([".", ",", "?", "-", ":", ""])
            for (_, char) in labels {
                assertTrue(allowed.contains(char), "label is allowed")
            }
            assertEqual(labels.count, 6, "exactly 6 labels")
        }

        suite("PunctuationService — overlength truncation")
        do {
            let modelDir = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache/fullstop-coreml/large")
            guard FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("tokenizer_compact.json").path) else {
                fail("tokenizer file missing")
                return
            }
            guard let tok = try? FullStopTokenizer(vocabDir: modelDir) else { fail("tokenizer init failed"); return }

            let longText = String(repeating: "the quick brown fox jumps over the lazy dog ", count: 20)
            let result = tok.encode(longText, maxLength: 256)
            assertTrue(result.inputIds.count == 256, "padded to maxLength=256")
        }

                // FullStop CoreML model availability check.
        // Integration tests that require async (model load, restore) run via
        // the Python verify step in export_fullstop.py instead.
        suite("PunctuationService — model artifact available")
        do {
            let modelDir = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".cache/fullstop-coreml/large")
            let mlpackage = modelDir.appendingPathComponent("fullstop-punctuation.mlpackage")
            let tokFile = modelDir.appendingPathComponent("tokenizer_compact.json")
            let cfgFile = modelDir.appendingPathComponent("config.json")

            let modelExists = FileManager.default.fileExists(atPath: mlpackage.path)
            let tokExists = FileManager.default.fileExists(atPath: tokFile.path)
            let cfgExists = FileManager.default.fileExists(atPath: cfgFile.path)

            assertTrue(modelExists, "mlpackage exists")
            assertTrue(tokExists, "tokenizer_compact.json exists")
            assertTrue(cfgExists, "config.json exists")

            if modelExists && tokExists {
                ok("all FullStop artifacts present")
            } else {
                fail("some FullStop artifacts missing — run Scripts/export_fullstop.py")
            }
        }

    }
}

let runner = TestRunner()
runner.runAll()
print("\n=== Results: \(runner.passed) passed, \(runner.failed) failed ===")
if runner.failed > 0 { exit(1) }
