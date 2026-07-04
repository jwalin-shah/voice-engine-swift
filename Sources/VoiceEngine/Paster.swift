import AppKit
import CoreGraphics
import Foundation

/// Injects transcribed text via simulated Cmd+V.
/// Uses CGEvent session tap — no Accessibility permission required.
public enum Paster {
    @discardableResult
    public static func paste(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        Thread.sleep(forTimeInterval: 0.02)
        let src = CGEventSource(stateID: .hidSystemState)
        guard let down = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: true),
              let up   = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: false) else {
            NSLog("[VoiceEngine] CGEvent creation failed")
            return false
        }
        down.flags = .maskCommand
        up.flags   = .maskCommand
        down.post(tap: .cgSessionEventTap)
        up.post(tap: .cgSessionEventTap)
        NSLog("[VoiceEngine] pasted \(text.count) chars via Cmd+V (session tap)")
        return true
    }
}
