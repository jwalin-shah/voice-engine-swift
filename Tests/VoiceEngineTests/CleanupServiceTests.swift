import Testing
@testable import VoiceEngine

@Suite("CleanupService") struct CleanupServiceTests {
    @Test func defaultMode() { let cs = CleanupService(); #expect(cs.mode == .fillerOnly) }
    @Test func modePersistence() { UserDefaults.standard.set(CleanupService.CleanupMode.disabled.rawValue, forKey: "cleanupMode"); let cs = CleanupService(); #expect(cs.mode == .disabled); UserDefaults.standard.removeObject(forKey: "cleanupMode") }
    @Test func disabledPassthrough() async { UserDefaults.standard.set(CleanupService.CleanupMode.disabled.rawValue, forKey: "cleanupMode"); let cs = CleanupService(); #expect(await cs.clean("hello world") == "hello world"); UserDefaults.standard.removeObject(forKey: "cleanupMode") }
    @Test func emptyInput() async { let cs = CleanupService(); #expect(await cs.clean("") == ""); #expect(await cs.clean("   ") == "   ") }
    @Test func fillerRemoval() async { let cs = CleanupService(); UserDefaults.standard.set(CleanupService.CleanupMode.fillerOnly.rawValue, forKey: "cleanupMode"); #expect(await cs.clean("hello um world") == "hello world"); UserDefaults.standard.removeObject(forKey: "cleanupMode") }
    @Test func checkAvailability() async { let cs = CleanupService(); #expect(await cs.checkAvailability() == true) }
    @Test func modeRawValues() { #expect(CleanupService.CleanupMode.disabled.rawValue == "Disabled"); #expect(CleanupService.CleanupMode.fillerOnly.rawValue == "Filler only"); #expect(CleanupService.CleanupMode.full.rawValue == "Full") }
    @Test func modeCodable() throws { let encoded = try JSONEncoder().encode(CleanupService.CleanupMode.full); let decoded = try JSONDecoder().decode(CleanupService.CleanupMode.self, from: encoded); #expect(decoded == .full) }
}
