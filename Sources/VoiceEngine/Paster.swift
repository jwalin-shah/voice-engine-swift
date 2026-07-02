import AppKit
import Carbon.HIToolbox
import CoreGraphics

/// Injects text into the focused application.
///
/// Strategy (tried in order):
///   1. voice-typer subprocess — compiled CGEvent binary, handles key mapping
///   2. CGEvent unicode — character-at-a-time, no Accessibility
///   3. AXUIElement setValue — direct, fast, needs Accessibility
///   4. Cmd+V via AppleScript — last resort, trashes clipboard
public enum Paster {
    @discardableResult
    public static func paste(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        if pasteViaVoiceTyper(text) { return true }
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

    /// CGEvent unicode — batch.
    private static func pasteViaCGEvent(_ text: String) -> Bool {
        let source = CGEventSource(stateID: .combinedSessionState)
        var chars = [UniChar]()
        chars.reserveCapacity(text.utf16.count)
        for scalar in text.unicodeScalars {
            let value = scalar.value
            if value < 0x20 && value != 0x0A && value != 0x09 { continue }
            if (0xD800...0xDFFF).contains(value) { continue }
            if value > UInt32(UInt16.max) {
                let leading = UniChar((value - 0x10000) >> 10) | 0xD800
                let trailing = UniChar((value - 0x10000) & 0x3FF) | 0xDC00
                chars.append(leading)
                chars.append(trailing)
            } else {
                chars.append(UniChar(value))
            }
        }
        guard !chars.isEmpty else { return false }
        let down = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true)
        let charCount = chars.count
        chars.withUnsafeMutableBufferPointer { ptr in
            down?.keyboardSetUnicodeString(stringLength: charCount, unicodeString: ptr.baseAddress!)
        }
        down?.post(tap: .cghidEventTap)
        let up = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false)
        up?.post(tap: .cghidEventTap)
        return true
    }

    /// voice-typer subprocess — compiled CGEvent binary.
    /// Uses UCKeyTranslate-based key mapping for better compatibility.
    /// Falls through to CGEvent if the binary is missing or fails.
    private static func pasteViaVoiceTyper(_ text: String) -> Bool {
        let binaryPath = "/Users/jwalinshah/local/bin/voice-typer"
        guard FileManager.default.fileExists(atPath: binaryPath) else { return false }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: binaryPath)
        process.arguments = [text]
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }
}
