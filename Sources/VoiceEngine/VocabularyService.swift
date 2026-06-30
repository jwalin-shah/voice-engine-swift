import AppKit

/// Manages user-defined vocabulary replacements and app-specific commands.
/// Stores mappings persistently in UserDefaults.
public final class VocabularyService {
    public static let shared = VocabularyService()

    private let defaults = UserDefaults.standard
    private let vocabKey = "customVocabulary"
    private let appCommandsKey = "appCommands"

    /// A single vocabulary entry: what the model hears → what to replace it with.
    public struct VocabEntry: Codable, Sendable {
        public let trigger: String   // "equestrian"
        public let replacement: String  // "Equestrian"
        public var isActive: Bool

        public init(trigger: String, replacement: String, isActive: Bool = true) {
            self.trigger = trigger
            self.replacement = replacement
            self.isActive = isActive
        }
    }

    /// App-specific command: when appName is frontmost, replace trigger.
    public struct AppCommand: Codable, Sendable {
        public let appName: String       // "Terminal", "Slack"
        public let bundleID: String      // "com.apple.Terminal"
        public let trigger: String
        public let replacement: String
        public var isActive: Bool

        public init(appName: String, bundleID: String, trigger: String, replacement: String, isActive: Bool = true) {
            self.appName = appName
            self.bundleID = bundleID
            self.trigger = trigger
            self.replacement = replacement
            self.isActive = isActive
        }
    }

    private init() {}

    // MARK: - Vocabulary

    public var vocabulary: [VocabEntry] {
        get { decodeArray(forKey: vocabKey) }
        set { encodeAndSave(newValue, forKey: vocabKey) }
    }

    public func addVocab(trigger: String, replacement: String) {
        var v = vocabulary
        // Update existing if trigger matches, else append
        if let idx = v.firstIndex(where: { $0.trigger.lowercased() == trigger.lowercased() }) {
            v[idx] = VocabEntry(trigger: trigger, replacement: replacement, isActive: v[idx].isActive)
        } else {
            v.append(VocabEntry(trigger: trigger, replacement: replacement))
        }
        vocabulary = v
    }

    public func applyVocabulary(to text: String) -> String {
        var result = text
        for entry in vocabulary where entry.isActive && !entry.trigger.isEmpty {
            // Case-insensitive replacement
            if let range = result.range(of: entry.trigger, options: [.caseInsensitive, .diacriticInsensitive]) {
                result.replaceSubrange(range, with: entry.replacement)
            }
        }
        return result
    }

    // MARK: - App Commands

    public var appCommands: [AppCommand] {
        get { decodeArray(forKey: appCommandsKey) }
        set { encodeAndSave(newValue, forKey: appCommandsKey) }
    }

    public func command(for bundleID: String, trigger: String) -> AppCommand? {
        appCommands.first { $0.bundleID == bundleID && $0.trigger.lowercased() == trigger.lowercased() && $0.isActive }
    }

    /// Apply both vocabulary AND app-specific commands for the given frontmost app.
    public func process(_ text: String, frontAppBundleID: String = "") -> String {
        var result = text

        // 1. App-specific commands (higher priority)
        for cmd in appCommands where cmd.isActive && cmd.bundleID == frontAppBundleID {
            if let range = result.range(of: cmd.trigger, options: [.caseInsensitive, .diacriticInsensitive]) {
                result.replaceSubrange(range, with: cmd.replacement)
            }
        }

        // 2. Global vocabulary
        result = applyVocabulary(to: result)

        return result
    }

    // MARK: - Helpers

    private func decodeArray<T: Codable>(forKey key: String) -> [T] {
        guard let data = defaults.data(forKey: key) else { return [] }
        return (try? JSONDecoder().decode([T].self, from: data)) ?? []
    }

    private func encodeAndSave<T: Codable>(_ value: [T], forKey key: String) {
        if let data = try? JSONEncoder().encode(value) {
            defaults.set(data, forKey: key)
        }
    }
}
