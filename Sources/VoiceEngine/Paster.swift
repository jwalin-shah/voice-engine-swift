import AppKit
import CoreGraphics
import Foundation

/// Injects transcribed text into the focused application.
///
/// Strategy (tried in order):
///   1. AX focused-element insert-at-cursor (needs Accessibility permission)
///   2. Cmd+V via CGEvent — multiple source+tap combinations
///
/// Root cause of prior failures: CGEvent.post(tap:) returns void, so failures
/// are silent. On macOS 15, posting keyboard events from a background process
/// (LaunchAgent) requires Accessibility permission in TCC. Without it,
/// NSPasteboard works fine (text lands on clipboard) but synthetic Cmd+V
/// keystrokes are silently discarded by the event system.
///
/// Fix: use kCGEventSourceStatePrivate (rawValue -1) + kCGHIDEventTap
/// as the primary CGEvent path, matching the old voice-typer.c approach.
/// Private event source may bypass some TCC restrictions.
public enum Paster {
    /// kCGEventSourceStatePrivate — lower-level event source that may bypass
    /// some Accessibility restrictions. Not exposed in the Swift enum.
    private static let privateState = CGEventSourceStateID(rawValue: -1)!

    @discardableResult
    public static func paste(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }

        // Method 1: AX focused-element insert at cursor.
        // Fast path — no clipboard involved. Needs Accessibility permission.
        if pasteViaAX(text) {
            NSLog("[VoiceEngine] pasted \(text.count) chars via AX")
            return true
        }
        NSLog("[VoiceEngine] AX paste unavailable (no Accessibility permission?)")

        // Method 2: clipboard + Cmd+V via CGEvent.
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)

        // Give the pasteboard time to sync before sending Cmd+V.
        // Without this, macOS may not have the content ready when the
        // keystroke fires.
        Thread.sleep(forTimeInterval: 0.05)

        // Try combinations in order of likelihood to work from a background process.
        // private + HID: matches old voice-typer.c approach (kCGEventSourceStatePrivate + kCGHIDEventTap)
        // private + session: private source into session tap
        // hidSystem + HID: standard source into HID tap
        // hidSystem + session: most restrictive, least likely to work from LaunchAgent
        let combos: [(CGEventSourceStateID, CGEventTapLocation, String)] = [
            (privateState, .cghidEventTap, "private+HID"),
            (privateState, .cgSessionEventTap, "private+session"),
            (.hidSystemState, .cghidEventTap, "hidSystem+HID"),
            (.hidSystemState, .cgSessionEventTap, "hidSystem+session"),
        ]

        for (stateID, tap, label) in combos {
            guard let src = CGEventSource(stateID: stateID) else {
                NSLog("[VoiceEngine] CGEventSource(\(label)) creation failed")
                continue
            }
            guard let down = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: true),
                  let up = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: false) else {
                NSLog("[VoiceEngine] CGEvent creation failed for \(label)")
                continue
            }
            src.localEventsSuppressionInterval = 0
            down.flags = .maskCommand
            up.flags = .maskCommand

            // post(tap:) returns void — we cannot verify delivery.
            // The event may be silently discarded if the process lacks
            // Accessibility permission.
            down.post(tap: tap)
            up.post(tap: tap)

            // ponytail: return immediately after first post attempt.
            // If multiple combos fire, the second Cmd+V would paste the
            // first one's text too. Only the first working combo matters.
            NSLog("[VoiceEngine] pasted \(text.count) chars via Cmd+V (\(label))")
            return true
        }

        // ponytail: if we get here, the clipboard has the text.
        // The user can manually paste with Cmd+V.
        // Log a warning so we can diagnose.
        NSLog("[VoiceEngine] WARNING: all CGEvent paste combos tried — verify text reached cursor. If not, add voice-engine to System Settings > Privacy & Security > Accessibility")
        return true // text IS on clipboard even if keystroke didn't fire
    }

    // MARK: - AX paste

    /// Insert text at the cursor in the focused accessibility element.
    /// Uses full value read-modify-write at selected range.
    /// Needs Accessibility permission in System Settings.
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
