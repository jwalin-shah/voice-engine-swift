import Testing
@testable import VoiceEngine
import Foundation

@Suite("DaemonIntegration") struct DaemonIntegrationTests {
    func mockScript() -> String { FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("projects/machine-scratch/voice-engine-swift/Tests/Resources/mock_daemon.py").path }
    @Test func launchAndPing() async { let d = DaemonService(); do { try await d.launch(scriptPath: mockScript()); #expect(d.isDaemonAvailable); let r = try await d.sendRequest(method: "ping"); #expect(r["status"] as? String == "ok"); d.shutdown() } catch { #expect(Bool(false), "Failed: \(error)") } }
    @Test func cleanupRequest() async { let d = DaemonService(); do { try await d.launch(scriptPath: mockScript()); let r = try await d.sendRequest(method: "cleanup", params: ["text": "hello"]); #expect(r["cleaned"] as? String == "hello [cleaned]"); d.shutdown() } catch { #expect(Bool(false), "Failed: \(error)") } }
    @Test func unknownMethod() async { let d = DaemonService(); do { try await d.launch(scriptPath: mockScript()); await #expect(throws: (any Error).self) { try await d.sendRequest(method: "unknown") }; d.shutdown() } catch { #expect(Bool(false), "Launch failed: \(error)") } }
}
