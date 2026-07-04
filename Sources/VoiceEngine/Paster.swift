import AppKit
import CoreGraphics
import Foundation

/// Injects transcribed text into the focused application.
/// Sets clipboard then fires a CGEvent Cmd+V directly — no subprocess, no latency.
/// Requires voice-engine to have Accessibility in System Settings → Privacy → Accessibility.
public enum Paster {
    @discardableResult
    public static func paste(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        return cmdV()
    }

    /// Sends Cmd+V directly via CGEvent — ~0ms overhead, no subprocess.
    private static func cmdV() -> Bool {
        let src = CGEventSource(stateID: .hidSystemState)
        // keycode 9 = v
        guard let down = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: true),
              let up   = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: false) else {
            NSLog("[VoiceEngine] CGEvent creation failed")
            return false
        }
        down.flags = .maskCommand
        up.flags   = .maskCommand
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
        NSLog("[VoiceEngine] pasted via CGEvent Cmd+V")
        return true
    }
}
