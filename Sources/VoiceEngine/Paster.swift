import AppKit
import CoreGraphics
import Foundation

/// Injects transcribed text via simulated Cmd+V.
/// Requires Accessibility: System Settings → Privacy → Accessibility → voice-engine
public enum Paster {
    @discardableResult
    public static func paste(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        // Small delay to let pasteboard flush before keystroke
        Thread.sleep(forTimeInterval: 0.02)
        let src = CGEventSource(stateID: .hidSystemState)
        guard let down = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: true),
              let up   = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: false) else {
            NSLog("[VoiceEngine] CGEvent creation failed — grant Accessibility permission")
            return false
        }
        down.flags = .maskCommand
        up.flags   = .maskCommand
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
        NSLog("[VoiceEngine] pasted \(text.count) chars via Cmd+V")
        return true
    }
}
