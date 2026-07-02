import Foundation

/// Text cleanup — filler word removal, pure Swift, no Python daemon.
public actor CleanupService {
    public enum CleanupMode: String, Codable, CaseIterable {
        case disabled = "Disabled"
        case fillerOnly = "Filler only"
        case full = "Full"
    }

    public init() {}

    public nonisolated var mode: CleanupMode {
        get {
            guard let raw = UserDefaults.standard.string(forKey: "cleanupMode"),
                  let mode = CleanupMode(rawValue: raw) else { return .fillerOnly }
            return mode
        }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: "cleanupMode") }
    }

    /// Always available — no daemon needed.
    public func checkAvailability() async -> Bool { true }

    /// Clean text: remove filler words if enabled.
    public func clean(_ text: String) -> String {
        guard mode != .disabled else { return text }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return text }
        if mode == .fillerOnly { return Self.removeFillers(trimmed) }
        // "full" mode is same as filler-only for now (daemon was never working)
        return Self.removeFillers(trimmed)
    }

    nonisolated static let fillerPatterns: [(NSRegularExpression, String)] = {
        let pairs: [(String, String)] = [
            ("\\bum\\b", ""), ("\\buh\\b", ""),
            ("\\blike\\b", ""), ("\\byou know\\b", ""),
            ("\\bi mean\\b", ""), ("\\bsort of\\b", ""),
            ("\\bkind of\\b", ""), ("\\bactually\\b", ""),
            ("\\bbasically\\b", ""), ("\\bliterally\\b", ""),
            ("\\bright\\b", ""), ("\\bso\\b", ""),
        ]
        return pairs.compactMap { (pattern, replacement) in
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return nil }
            return (regex, replacement)
        }
    }()

    nonisolated static func removeFillers(_ text: String) -> String {
        let ns = text as NSString
        var result = text
        for (regex, replacement) in fillerPatterns {
            result = regex.stringByReplacingMatches(in: result, range: NSRange(location: 0, length: (result as NSString).length), withTemplate: replacement)
        }
        // Collapse multiple spaces
        result = result.replacingOccurrences(of: " +", with: " ", options: .regularExpression)
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
