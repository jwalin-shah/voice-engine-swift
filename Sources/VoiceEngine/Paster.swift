import AppKit
import Foundation

/// Injects transcribed text into the focused application.
///
/// Strategy (tried in order):
///   1. AX focused-element insert (needs Accessibility permission — instant, no clipboard)
///   2. osascript Cmd+V via System Events (works NOW — System Events has Accessibility)
///   3. CGEvent Cmd+V (fast path for when voice-engine itself gets Accessibility)
///
/// ponytail: osascript adds ~80ms vs ~50ms for CGEvent — both invisible next to
/// the 32ms ML inference. Process spawn is worth the reliability.
public enum Paster {
    @discardableResult
    public static func paste(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }

        // Snapshot the frontmost app NOW — before clipboard ops or anything
        // that might cause VoiceEngine to steal focus. VoiceEngine is
        // .accessory policy so it shouldn't be frontmost, but if focus
        // drifted during recording/transcription, we capture the real target.
        let targetApp = NSWorkspace.shared.frontmostApplication

        // Method 1: AX direct insert at cursor. No clipboard, fastest path.
        // Works when voice-engine has Accessibility permission.
        if pasteViaAX(text) {
            NSLog("[VoiceEngine] pasted \(text.count) chars via AX")
            return true
        }

        // Clipboard for methods 2 and 3.
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        Thread.sleep(forTimeInterval: 0.05)  // ponytail: 50ms for pasteboard commit

        // Bring the target app back to front before pasting.
        // Belt and suspenders: if focus hasn't shifted, activate is a no-op.
        if let app = targetApp {
            app.activate()
            Thread.sleep(forTimeInterval: 0.05)  // let activation settle
        }

        // Method 2: osascript tells System Events to type Cmd+V.
        // System Events has Accessibility permission — this works NOW,
        // no grant needed for voice-engine itself.
        if pasteViaAppleScript() {
            NSLog("[VoiceEngine] pasted \(text.count) chars via osascript Cmd+V")
            return true
        }

        // Method 3: CGEvent Cmd+V — fast path for when voice-engine
        // has been added to Accessibility (after AXIsProcessTrustedWithOptions prompt).
        if pasteViaCGEvent() {
            NSLog("[VoiceEngine] pasted \(text.count) chars via CGEvent Cmd+V")
            return true
        }

        // Text is on clipboard even if all methods failed.
        // User can paste manually with Cmd+V.
        NSLog("[VoiceEngine] WARNING: all paste methods failed. Text is on clipboard — add voice-engine to System Settings > Privacy & Security > Accessibility for direct paste.")
        return true
    }

    // MARK: - AppleScript paste (System Events, works NOW)

    private static func pasteViaAppleScript() -> Bool {
        let task = Process()
        task.launchPath = "/usr/bin/osascript"
        task.arguments = ["-e", "tell application \"System Events\" to keystroke \"v\" using command down"]
        task.standardOutput = FileHandle.nullDevice
        task.standardError = FileHandle.nullDevice
        task.launch()
        task.waitUntilExit()
        return task.terminationStatus == 0
    }

    // MARK: - CGEvent paste (fast path with Accessibility)

    private static func pasteViaCGEvent() -> Bool {
        guard let src = CGEventSource(stateID: .hidSystemState),
              let down = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: true),
              let up = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: false) else {
            return false
        }
        src.localEventsSuppressionInterval = 0
        down.flags = .maskCommand
        up.flags = .maskCommand
        down.post(tap: .cgSessionEventTap)
        up.post(tap: .cgSessionEventTap)
        return true
    }

    // MARK: - AX paste (best path, needs Accessibility)

    private static func pasteViaAX(_ text: String) -> Bool {
        guard let element = focusedElement(),
              let originalText = elementText(element),
              let selectedRange = selectedRange(element) else { return false }

        let nsText = NSMutableString(string: originalText)
        let range = NSRange(location: selectedRange.location, length: selectedRange.length)
        guard range.location >= 0,
              range.length >= 0,
              range.location <= nsText.length,
              range.location + range.length <= nsText.length else { return false }

        nsText.replaceCharacters(in: range, with: text)
        guard AXUIElementSetAttributeValue(element, kAXValueAttribute as CFString, nsText as CFTypeRef) == .success else { return false }

        var cursorRange = CFRange(location: selectedRange.location + (text as NSString).length, length: 0)
        if let axRange = AXValueCreate(.cfRange, &cursorRange) {
            AXUIElementSetAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, axRange)
        }
        return true
    }

    // MARK: - AX helpers

    private static func focusedElement() -> AXUIElement? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let appRef = AXUIElementCreateApplication(app.processIdentifier)
        var focusedRef: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(appRef, kAXFocusedUIElementAttribute as CFString, &focusedRef)
        guard result == .success, let focusedRef else { return nil }
        return (focusedRef as! AXUIElement)
    }

    private static func elementText(_ element: AXUIElement) -> String? {
        var valueRef: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &valueRef)
        guard result == .success else { return nil }
        return valueRef as? String
    }

    private static func selectedRange(_ element: AXUIElement) -> CFRange? {
        var rangeRef: CFTypeRef?
        let result = AXUIElementCopyAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, &rangeRef)
        guard result == .success, let rangeRef else { return nil }
        var range = CFRange()
        guard AXValueGetValue((rangeRef as! AXValue), .cfRange, &range) else { return nil }
        return range
    }
}
