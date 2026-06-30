import Testing
@testable import VoiceEngine

@Suite("DaemonService") struct DaemonServiceTests {
    @Test func initialState() { let d = DaemonService(); #expect(d.status == .notLaunched); #expect(d.isDaemonAvailable == false) }
    @Test func errorDescriptions() { #expect(DaemonService.DaemonError.notRunning.errorDescription != nil); #expect(DaemonService.DaemonError.timeout.errorDescription != nil); #expect(DaemonService.DaemonError.rpcError("x").errorDescription != nil); #expect(DaemonService.DaemonError.pythonNotFound.errorDescription != nil); #expect(DaemonService.DaemonError.processExited(1).errorDescription != nil) }
    @Test func sendWithoutProcess() async { let d = DaemonService(); await #expect(throws: (any Error).self) { try await d.sendRequest(method: "ping") } }
    @Test func shutdownNoProcess() { let d = DaemonService(); d.shutdown(); #expect(d.isDaemonAvailable == false) }
}
