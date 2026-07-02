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
        suite("MoonshineEngine.chunkRanges")
        assertEqual(MoonshineEngine.chunkRanges(sampleCount: 0).count, 0, "empty audio has no chunks")
        assertEqual(MoonshineEngine.chunkRanges(sampleCount: 160000), [0..<160000], "exact encoder window is one chunk")
        assertEqual(MoonshineEngine.chunkRanges(sampleCount: 168000), [0..<160000], "sub-second tail after overlap is skipped")
        assertEqual(MoonshineEngine.chunkRanges(sampleCount: 176000), [0..<160000, 128000..<176000], "one-second tail after overlap is kept")
        assertEqual(MoonshineEngine.chunkRanges(sampleCount: 288000), [0..<160000, 128000..<288000], "eighteen seconds uses two chunks")
    }
}

let runner = TestRunner()
runner.runAll()
print("\n=== Results: \(runner.passed) passed, \(runner.failed) failed ===")
if runner.failed > 0 { exit(1) }
