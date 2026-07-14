import AppKit
import CoreGraphics

/// Voice commands that map to keystrokes — no AX text manipulation.
///
/// Pattern: commands are either pure keystrokes ("undo" → Cmd+Z) or suffix-stripped
/// from dictation text ("hello world press enter" → paste "hello world", send Return).
/// All execution goes through CGEvent keystroke simulation — same mechanism as
/// Paster's tier 3 — so reliability matches paste reliability.
public enum CommandParser {

    // MARK: - VoiceCommand

    public enum VoiceCommand: Equatable, Sendable {
        case pressEnter       // Return keystroke
        case newParagraph     // Return × 2
        case tab              // Tab keystroke
        case undo             // Cmd+Z
    }

    // MARK: - Public API

    /// Parse the entire utterance as a pure command.
    /// Returns nil if the text does not match any command pattern.
    public static func parse(_ text: String) -> VoiceCommand? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !trimmed.isEmpty else { return nil }

        switch trimmed {
        case "undo":               return .undo
        case "new line", "newline": return .pressEnter
        case "new paragraph":       return .newParagraph
        case "tab":                 return .tab
        case "press enter":         return .pressEnter
        default:                    return nil
        }
    }

    /// Extract a command suffix from the end of text.
    /// The suffix must be a whole-word match at the end (preceded by whitespace).
    /// Returns the prefix text (to paste) and the command (to execute as keystroke).
    public static func extractCommand(from text: String) -> (prefix: String, command: VoiceCommand)? {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return nil }
        let lowered = trimmed.lowercased()

        // ponytail: longest-first to avoid "new line" matching inside "new paragraph"
        let patterns: [(keyword: String, command: VoiceCommand)] = [
            ("new paragraph", .newParagraph),
            ("press enter",   .pressEnter),
            ("new line",      .pressEnter),
            ("newline",       .pressEnter),
            ("undo",          .undo),
            ("tab",           .tab),
        ]

        for (keyword, command) in patterns {
            if lowered == keyword { continue } // pure command — handled by parse()
            if lowered.hasSuffix(keyword) {
                let prefixEnd = trimmed.index(trimmed.endIndex, offsetBy: -keyword.count)
                let charBefore = trimmed[trimmed.index(before: prefixEnd)]
                guard charBefore.isWhitespace else { continue }
                let prefix = String(trimmed[..<prefixEnd]).trimmingCharacters(in: .whitespaces)
                return (prefix, command)
            }
        }

        return nil
    }

    // MARK: - Execute

    /// Execute a command via CGEvent keystroke simulation.
    @discardableResult
    public static func execute(_ command: VoiceCommand) -> Bool {
        switch command {
        case .pressEnter:
            return postKey(keyCode: 36) // kVK_Return
        case .newParagraph:
            return postKey(keyCode: 36) && postKey(keyCode: 36)
        case .tab:
            return postKey(keyCode: 48) // kVK_Tab
        case .undo:
            return postKey(keyCode: 6, flags: .maskCommand) // kVK_ANSI_Z
        }
    }

    // MARK: - Keystroke helpers

    private static func postKey(keyCode: UInt16, flags: CGEventFlags = []) -> Bool {
        guard let src = CGEventSource(stateID: .combinedSessionState) else { return false }
        guard let down = CGEvent(keyboardEventSource: src, virtualKey: keyCode, keyDown: true),
              let up   = CGEvent(keyboardEventSource: src, virtualKey: keyCode, keyDown: false) else {
            return false
        }
        down.flags = flags
        up.flags   = flags
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
        return true
    }
}
