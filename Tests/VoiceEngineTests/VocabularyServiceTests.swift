import Testing
@testable import VoiceEngine

@Suite("VocabularyService") struct VocabularyServiceTests {
    init() { UserDefaults.standard.removeObject(forKey: "customVocabulary"); UserDefaults.standard.removeObject(forKey: "appCommands") }
    @Test func emptyVocab() { #expect(VocabularyService.shared.vocabulary.isEmpty); #expect(VocabularyService.shared.process("hello world") == "hello world") }
    @Test func singleEntry() { VocabularyService.shared.addVocab(trigger: "equestrian", replacement: "Equestrian"); #expect(VocabularyService.shared.applyVocabulary(to: "the equestrian center") == "the Equestrian center"); VocabularyService.shared.vocabulary = [] }
    @Test func inactiveEntry() { var v = VocabularyService.shared.vocabulary; v.append(VocabularyService.VocabEntry(trigger: "test", replacement: "TEST", isActive: false)); VocabularyService.shared.vocabulary = v; #expect(VocabularyService.shared.applyVocabulary(to: "test") == "test"); VocabularyService.shared.vocabulary = [] }
    @Test func persistence() { VocabularyService.shared.addVocab(trigger: "persist", replacement: "Persist"); let loaded = VocabularyService.shared.vocabulary; #expect(loaded.count == 1); #expect(loaded[0].trigger == "persist"); VocabularyService.shared.vocabulary = [] }
    @Test func appCommands() { var cmds = VocabularyService.shared.appCommands; cmds.append(VocabularyService.AppCommand(appName: "Test", bundleID: "com.test.app", trigger: "apphello", replacement: "APPHELLO")); VocabularyService.shared.appCommands = cmds; #expect(VocabularyService.shared.process("say apphello", frontAppBundleID: "com.test.app") == "say APPHELLO"); VocabularyService.shared.appCommands = [] }
    @Test func appCommandsWrongBundle() { var cmds = VocabularyService.shared.appCommands; cmds.append(VocabularyService.AppCommand(appName: "Test", bundleID: "com.other.app", trigger: "apphello", replacement: "APPHELLO")); VocabularyService.shared.appCommands = cmds; #expect(VocabularyService.shared.process("say apphello", frontAppBundleID: "com.test.app") == "say apphello"); VocabularyService.shared.appCommands = [] }
}
