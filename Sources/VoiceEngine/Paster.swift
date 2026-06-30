import AppKit
import Carbon.HIToolbox
import CoreGraphics

/// Injects text into the focused application.
///
/// Strategy (tried in order):
///   1. CGEvent unicode — character-at-a-time, reliable, no clipboard, no Accessibility
///   2. AXUIElement setValue — direct, fast, needs Accessibility
///   3. Cmd+V via AppleScript — last resort, trashes clipboard
public enum Paster {
    @discardableResult
    public static func paste(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        if pasteViaCGEvent(text) { return true }
        if pasteViaAX(text) { return true }
        return pasteViaASE(text)
    }

    /// AppleScript Cmd+V paste.
    private static func pasteViaASE(_ text: String) -> Bool {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
        let s = "tell application \"System Events\" to keystroke \"v\" using {command down}"
        var error: NSDictionary?
        return NSAppleScript(source: s)?.executeAndReturnError(&error) != nil
    }

    /// AX focused element direct set.
    private static func pasteViaAX(_ text: String) -> Bool {
        guard let app = NSWorkspace.shared.frontmostApplication else { return false }
        let pid = app.processIdentifier
        let appRef = AXUIElementCreateApplication(pid)
        var focusedRef: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(appRef, kAXFocusedUIElementAttribute as CFString, &focusedRef)
        guard result == .success, let focused = focusedRef else { return false }
        let element = focused as! AXUIElement
        return AXUIElementSetAttributeValue(element, kAXValueAttribute as CFString, text as CFTypeRef) == .success
    }

    /// CGEvent unicode — character-at-a-time.
    private static func pasteViaCGEvent(_ text: String) -> Bool {
        let source = CGEventSource(stateID: .combinedSessionState)
        for scalar in text.unicodeScalars {
            let value = scalar.value
            if value < 0x20 && value != 0x0A && value != 0x09 { continue }
            if (0xD800...0xDFFF).contains(value) { continue }
            let down = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true)
            var uniChar = UniChar(value)
            down?.keyboardSetUnicodeString(stringLength: 1, unicodeString: &uniChar)
            down?.post(tap: .cghidEventTap)
            let up = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false)
            up?.post(tap: .cghidEventTap)
        }
        return true
    }
}
