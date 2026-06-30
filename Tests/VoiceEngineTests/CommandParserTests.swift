import Testing
@testable import VoiceEngine

@Test("undo") func undo() { #expect(CommandParser.parse("undo") == .undo) }
@Test("new line") func newLine() { #expect(CommandParser.parse("new line") == .newLine); #expect(CommandParser.parse("newline") == .newLine) }
@Test("new paragraph") func newParagraph() { #expect(CommandParser.parse("new paragraph") == .newParagraph) }
@Test("tab") func tab() { #expect(CommandParser.parse("tab") == .tab) }
@Test("delete") func delete() { #expect(CommandParser.parse("delete that") == .deleteThat); #expect(CommandParser.parse("delete this") == .deleteThat) }
@Test("capitalize") func capitalize() { #expect(CommandParser.parse("capitalize that") == .capitalizeThat); #expect(CommandParser.parse("uppercase that") == .capitalizeThat) }
@Test("lowercase") func lowercase() { #expect(CommandParser.parse("lowercase that") == .lowercaseThat) }
@Test("select") func select() { #expect(CommandParser.parse("select hello world") == .select("hello world")); #expect(CommandParser.parse("SELECT HELLO") == .select("SELECT HELLO")) }
@Test("select edge") func selectEdge() { #expect(CommandParser.parse("select ") == nil); #expect(CommandParser.parse("select") == nil) }
@Test("replace") func replace() { #expect(CommandParser.parse("replace foo with bar") == .replace("foo", "bar")); #expect(CommandParser.parse("replace a with b c") == .replace("a", "b c")) }
@Test("replace edge") func replaceEdge() { #expect(CommandParser.parse("replace foo bar") == nil); #expect(CommandParser.parse("replace  with ") == nil) }
@Test("no match") func noMatch() { #expect(CommandParser.parse("hello world") == nil); #expect(CommandParser.parse("") == nil); #expect(CommandParser.parse("   ") == nil) }
@Test("padded") func padded() { #expect(CommandParser.parse("  undo  ") == .undo) }

@Suite struct ExtractTests {
    @Test func suffixUndo() { let r = CommandParser.extractCommand(from: "hello world undo"); #expect(r?.prefix == "hello world"); #expect(r?.command == .undo) }
    @Test func suffixNewLine() { #expect(CommandParser.extractCommand(from: "hello new line")?.command == .newLine) }
    @Test func suffixCap() { #expect(CommandParser.extractCommand(from: "hello world capitalize that")?.command == .capitalizeThat) }
    @Test func suffixSelect() { let r = CommandParser.extractCommand(from: "hello world select foo"); #expect(r?.prefix == "hello world"); #expect(r?.command == .select("foo")) }
    @Test func suffixReplace() { #expect(CommandParser.extractCommand(from: "hello world replace old with new")?.command == .replace("old", "new")) }
    @Test func pureSkipped() { #expect(CommandParser.extractCommand(from: "undo") == nil); #expect(CommandParser.extractCommand(from: "new line") == nil) }
    @Test func noBoundary() { #expect(CommandParser.extractCommand(from: "helloundo") == nil); #expect(CommandParser.extractCommand(from: "hello_undo") == nil) }
    @Test func emptyInput() { #expect(CommandParser.extractCommand(from: "") == nil) }
    @Test func miscSuffixes() { #expect(CommandParser.extractCommand(from: "hello world delete that")?.command == .deleteThat); #expect(CommandParser.extractCommand(from: "hello lowercase that")?.command == .lowercaseThat); #expect(CommandParser.extractCommand(from: "hello uppercase that")?.command == .capitalizeThat) }
    @Test func tabParagraph() { #expect(CommandParser.extractCommand(from: "hello tab")?.command == .tab); #expect(CommandParser.extractCommand(from: "hello new paragraph")?.command == .newParagraph) }
}
@Suite struct EquatableTests {
    @Test func equality() { #expect(CommandParser.VoiceCommand.select("a") == .select("a")); #expect(CommandParser.VoiceCommand.select("a") != .select("b")); #expect(CommandParser.VoiceCommand.undo == .undo); #expect(CommandParser.VoiceCommand.undo != .newLine) }
}
