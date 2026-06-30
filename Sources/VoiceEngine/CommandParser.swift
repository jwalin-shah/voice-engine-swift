import AppKit
import CoreGraphics
import Carbon.HIToolbox

public enum CommandParser {

    // MARK: - VoiceCommand
    public enum VoiceCommand: Equatable, Sendable {
        case select(String)
        case capitalizeThat
        case lowercaseThat
        case deleteThat
        case replace(String, String)
        case newLine
        case newParagraph
        case tab
        case undo
    }

    // MARK: - Public API

    /// Parse the entire utterance as a pure command.
    /// Returns nil if the text does not match any command pattern.
    public static func parse(_ text: String) -> VoiceCommand? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let lowered = trimmed.lowercased()

        // replace <old> with <new> (most specific first)
        if lowered.hasPrefix("replace "), lowered.count > 16 {
            let afterReplace = String(trimmed[trimmed.index(trimmed.startIndex, offsetBy: 8)...])
            if let withRange = afterReplace.lowercased().range(of: " with ") {
                let old = String(afterReplace[..<withRange.lowerBound]).trimmingCharacters(in: .whitespaces)
                let new = String(afterReplace[withRange.upperBound...]).trimmingCharacters(in: .whitespaces)
                if !old.isEmpty && !new.isEmpty {
                    return .replace(old, new)
                }
            }
        }

        // select <text>
        if lowered.hasPrefix("select ") {
            let target = String(trimmed[trimmed.index(trimmed.startIndex, offsetBy: 7)...])
                .trimmingCharacters(in: .whitespaces)
            if !target.isEmpty {
                return .select(target)
            }
        }

        // Exact matches
        switch lowered {
        case "undo": return .undo
        case "new line", "newline": return .newLine
        case "new paragraph": return .newParagraph
        case "tab": return .tab
        case "delete that", "delete this": return .deleteThat
        case "capitalize that", "uppercase that": return .capitalizeThat
        case "lowercase that": return .lowercaseThat
        default: break
        }

        return nil
    }

    /// Extract a command suffix from the end of text.
    /// The suffix must be a whole-word match at the end (preceded by whitespace).
    public static func extractCommand(from text: String) -> (prefix: String, command: VoiceCommand)? {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return nil }
        let lowered = trimmed.lowercased()

        // Simple suffix patterns (longest first to avoid partial matches)
        let patterns: [(keyword: String, command: VoiceCommand)] = [
            ("new paragraph", .newParagraph),
            ("capitalize that", .capitalizeThat),
            ("uppercase that", .capitalizeThat),
            ("lowercase that", .lowercaseThat),
            ("delete that", .deleteThat),
            ("delete this", .deleteThat),
            ("new line", .newLine),
            ("newline", .newLine),
            ("undo", .undo),
            ("tab", .tab),
        ]

        for (keyword, command) in patterns {
            if lowered == keyword { continue } // pure command -- handled by parse()
            if lowered.hasSuffix(keyword) {
                let prefixEnd = trimmed.index(trimmed.endIndex, offsetBy: -keyword.count)
                let charBefore = trimmed[trimmed.index(before: prefixEnd)]
                guard charBefore.isWhitespace else { continue }
                let prefix = String(trimmed[..<prefixEnd]).trimmingCharacters(in: .whitespaces)
                return (prefix, command)
            }
        }

        // Parameterized suffix: "prefix select <text>"
        if let selRange = lowered.range(of: " select ", options: .backwards) {
            let target = String(trimmed[selRange.upperBound...]).trimmingCharacters(in: .whitespaces)
            if !target.isEmpty {
                let prefix = String(trimmed[..<selRange.lowerBound]).trimmingCharacters(in: .whitespaces)
                return (prefix, .select(target))
            }
        }

        // Parameterized suffix: "prefix replace <old> with <new>"
        if let repRange = lowered.range(of: " replace ", options: .backwards) {
            let afterReplace = String(trimmed[repRange.upperBound...]).trimmingCharacters(in: .whitespaces)
            if let withRange = afterReplace.lowercased().range(of: " with ") {
                let old = String(afterReplace[..<withRange.lowerBound]).trimmingCharacters(in: .whitespaces)
                let new = String(afterReplace[withRange.upperBound...]).trimmingCharacters(in: .whitespaces)
                if !old.isEmpty && !new.isEmpty {
                    let prefix = String(trimmed[..<repRange.lowerBound]).trimmingCharacters(in: .whitespaces)
                    return (prefix, .replace(old, new))
                }
            }
        }

        return nil
    }

    // MARK: - Execute

    @discardableResult
    public static func execute(_ command: VoiceCommand) -> Bool {
        switch command {
        case .select(let target): return executeSelect(target)
        case .capitalizeThat: return executeTransform { $0.localizedCapitalized }
        case .lowercaseThat: return executeTransform { $0.localizedLowercase }
        case .deleteThat: return executeDeleteSelection()
        case .replace(let old, let new): return executeReplace(old: old, new: new)
        case .newLine: return executeInsertText("\n")
        case .newParagraph: return executeInsertText("\n\n")
        case .tab: return executeInsertText("\t")
        case .undo: return executeUndo()
        }
    }

    // MARK: - AX Helpers

    private static func getFocusedElement() -> AXUIElement? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let appRef = AXUIElementCreateApplication(app.processIdentifier)
        var focusedRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(appRef, kAXFocusedUIElementAttribute as CFString, &focusedRef) == .success,
              let focused = focusedRef else { return nil }
        return (focused as! AXUIElement)
    }

    private static func getElementText(_ element: AXUIElement) -> String? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &value) == .success,
              let string = value as? String else { return nil }
        return string
    }

    @discardableResult
    private static func setElementText(_ element: AXUIElement, _ text: String) -> Bool {
        AXUIElementSetAttributeValue(element, kAXValueAttribute as CFString, text as CFTypeRef) == .success
    }

    private static func getSelectedRange(_ element: AXUIElement) -> CFRange? {
        var rangeRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, &rangeRef) == .success,
              let axValue = rangeRef else { return nil }
        var range = CFRange(location: 0, length: 0)
        guard AXValueGetValue(axValue as! AXValue, .cfRange, &range) else { return nil }
        return range
    }

    @discardableResult
    private static func setSelectedRange(_ element: AXUIElement, _ range: CFRange) -> Bool {
        var r = range
        let axValue = AXValueCreate(.cfRange, &r)!
        return AXUIElementSetAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, axValue) == .success
    }

    // MARK: - Execution

    /// select <text>
    private static func executeSelect(_ target: String) -> Bool {
        guard let element = getFocusedElement(),
              let fullText = getElementText(element) else { return false }
        // Case-insensitive search for first occurrence
        guard let range = fullText.lowercased().range(of: target.lowercased()) else { return false }
        let location = fullText.distance(from: fullText.startIndex, to: range.lowerBound)
        let length = fullText.distance(from: range.lowerBound, to: range.upperBound)
        return setSelectedRange(element, CFRange(location: location, length: length))
    }

    /// capitalize/lowercase selected text
    private static func executeTransform(with transform: (String) -> String) -> Bool {
        guard let element = getFocusedElement(),
              let fullText = getElementText(element),
              let selRange = getSelectedRange(element),
              selRange.length > 0 else { return false }
        let startIdx = fullText.index(fullText.startIndex, offsetBy: selRange.location)
        let endIdx = fullText.index(startIdx, offsetBy: selRange.length)
        let selected = String(fullText[startIdx..<endIdx])
        let transformed = transform(selected)
        let newText = fullText.replacingCharacters(in: startIdx..<endIdx, with: transformed)
        guard setElementText(element, newText) else { return false }
        // Restore selection to highlight transformed text
        let newLen = (transformed as NSString).length
        return setSelectedRange(element, CFRange(location: selRange.location, length: newLen))
    }

    /// delete that/this
    private static func executeDeleteSelection() -> Bool {
        guard let element = getFocusedElement(),
              let fullText = getElementText(element),
              let selRange = getSelectedRange(element),
              selRange.length > 0 else { return false }
        let startIdx = fullText.index(fullText.startIndex, offsetBy: selRange.location)
        let endIdx = fullText.index(startIdx, offsetBy: selRange.length)
        let newText = String(fullText[..<startIdx] + fullText[endIdx...])
        guard setElementText(element, newText) else { return false }
        // Cursor at deletion point
        return setSelectedRange(element, CFRange(location: selRange.location, length: 0))
    }

    /// replace <old> with <new>
    private static func executeReplace(old: String, new: String) -> Bool {
        guard let element = getFocusedElement(),
              let fullText = getElementText(element) else { return false }
        guard let matchRange = fullText.lowercased().range(of: old.lowercased()) else { return false }
        let result = fullText.replacingCharacters(in: matchRange, with: new)
        guard setElementText(element, result) else { return false }
        // Cursor after replacement
        let location = fullText.distance(from: fullText.startIndex, to: matchRange.lowerBound)
        let cursorPos = location + (new as NSString).length
        return setSelectedRange(element, CFRange(location: cursorPos, length: 0))
    }

    /// insert text at cursor (newline, paragraph, tab)
    private static func executeInsertText(_ text: String) -> Bool {
        guard let element = getFocusedElement(),
              let fullText = getElementText(element),
              let selRange = getSelectedRange(element) else { return false }
        let insertionPoint = selRange.location
        let startIdx = fullText.index(fullText.startIndex, offsetBy: insertionPoint)
        let newText = String(fullText[..<startIdx]) + text + String(fullText[startIdx...])
        guard setElementText(element, newText) else { return false }
        let cursorPos = insertionPoint + (text as NSString).length
        return setSelectedRange(element, CFRange(location: cursorPos, length: 0))
    }

    /// undo via Cmd+Z
    private static func executeUndo() -> Bool {
        guard let source = CGEventSource(stateID: .combinedSessionState) else { return false }
        let keyCode: UInt16 = 6 // kVK_ANSI_Z
        guard let down = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: true) else { return false }
        down.flags = .maskCommand
        down.post(tap: .cghidEventTap)
        guard let up = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: false) else { return false }
        up.flags = .maskCommand
        up.post(tap: .cghidEventTap)
        return true
    }
}
