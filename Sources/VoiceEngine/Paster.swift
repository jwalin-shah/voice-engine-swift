import AppKit
import CoreGraphics
import Foundation

/// Injects transcribed text into the focused application.
/// Uses AX insertion when the focused element exposes text editing attributes,
/// then falls back to clipboard-backed Cmd+V.
public enum Paster {
    @discardableResult
    public static func paste(_ text: String) -> Bool {
        guard !text.isEmpty else { return false }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)

        if pasteViaAX(text) {
            NSLog("[VoiceEngine] pasted \(text.count) chars via AX focused element")
            return true
        }

        // ponytail: no sleep needed — localEventsSuppressionInterval=0 in postCommandV
        // prevents macOS from suppressing the synthetic Cmd+V.
        if postCommandV(tap: .cgSessionEventTap, label: "session") {
            return true
        }
        return postCommandV(tap: .cghidEventTap, label: "hid")
    }

    private static func postCommandV(tap: CGEventTapLocation, label: String) -> Bool {
        guard let src = CGEventSource(stateID: .hidSystemState),
              let down = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: true),
              let up = CGEvent(keyboardEventSource: src, virtualKey: 9, keyDown: false) else {
            NSLog("[VoiceEngine] CGEvent creation failed")
            return false
        }
        src.localEventsSuppressionInterval = 0
        down.flags = .maskCommand
        up.flags = .maskCommand
        // ponytail: post back-to-back — real keypresses don't pause between down and up.
        // localEventsSuppressionInterval=0 means macOS won't suppress the synthetic event.
        down.post(tap: tap)
        up.post(tap: tap)
        NSLog("[VoiceEngine] pasted via Cmd+V (\(label) tap)")
        return true
    }

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
        guard setValue(String(nsText), for: element) else { return false }

        var cursorRange = CFRange(location: selectedRange.location + (text as NSString).length, length: 0)
        guard let axRange = AXValueCreate(.cfRange, &cursorRange) else { return true }
        _ = AXUIElementSetAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, axRange)
        return true
    }

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

    private static func setValue(_ value: String, for element: AXUIElement) -> Bool {
        AXUIElementSetAttributeValue(element, kAXValueAttribute as CFString, value as CFTypeRef) == .success
    }
}
