import AppKit
import Foundation

/// Injects transcribed text into the focused application.
///
/// Single fast path: clipboard + CGEvent Cmd+V. No Accessibility permission
/// required. VoiceEngine is .accessory policy so it never steals focus — the
/// target app is always frontmost when paste runs.
///
/// ~1ms: setString blocks synchronously (~25µs median), CGEvent posts into
/// the target app's event stream (~1ms). No sleeps. No process spawn. No AX.
public enum Paster {
    @discardableResult
    public static func paste(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }

        // Save current clipboard contents so we can restore if paste fails.
        let savedString = NSPasteboard.general.string(forType: .string)

        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)

        guard let src = CGEventSource(stateID: .hidSystemState),
              let down = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: true),
              let up = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: false) else {
            // Paste failed — restore original clipboard contents.
            NSPasteboard.general.clearContents()
            if let saved = savedString {
                NSPasteboard.general.setString(saved, forType: .string)
            }
            NSLog("[VoiceEngine] WARNING: CGEvent failed, clipboard restored")
            return false
        }
        src.localEventsSuppressionInterval = 0
        down.flags = .maskCommand
        up.flags = .maskCommand
        down.post(tap: .cgSessionEventTap)
        up.post(tap: .cgSessionEventTap)
        return true
    }
}