import Foundation

public actor CleanupService {
    public enum CleanupMode: String, Codable, CaseIterable {
        case disabled = "Disabled"
        case fillerOnly = "Filler only"
        case full = "Full"
    }

    private let daemonService: DaemonService
    private(set) var isAvailable = false

    public init(daemon: DaemonService) {
        self.daemonService = daemon
    }

    public nonisolated var mode: CleanupMode {
        get {
            guard let raw = UserDefaults.standard.string(forKey: "cleanupMode"),
                  let mode = CleanupMode(rawValue: raw) else { return .full }
            return mode
        }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: "cleanupMode") }
    }

    func setMode(_ mode: CleanupMode) { self.mode = mode }

    public func checkAvailability() async -> Bool {
        do {
            let result = try await daemonService.sendRequest(method: "ping", params: [:])
            isAvailable = (result["model_loaded"] as? Bool) ?? false
            return isAvailable
        } catch {
            isAvailable = false; return false
        }
    }

    public func clean(_ text: String) async -> String {
        guard mode != .disabled else { return text }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return text }
        let daemonMode: String = mode == .fillerOnly ? "filler_only" : "full"
        do {
            let result = try await daemonService.sendRequest(method: "cleanup", params: ["text": text, "mode": daemonMode])
            return (result["cleaned"] as? String ?? text).trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            NSLog("[CleanupService] Cleanup failed: \(error.localizedDescription)")
            return text
        }
    }
}
