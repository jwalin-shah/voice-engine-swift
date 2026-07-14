import Foundation

/// Deterministic rule-based capitalization — pure, stateless, no model.
///
/// Three rules applied in order:
/// 1. Sentence-start capitalization — first letter of text and first letter
///    after each . ! ?.
/// 2. Proper noun dictionary — case-insensitive whole-word lookup for known
///    entities (GitHub, macOS, JSON, etc.). Runs after sentence-start so
///    mixed-casing entries can override incorrect initial caps.
/// 3. Standalone pronoun "i" → "I" — word-boundary match, does not touch
///    "i" inside other words (it, in, iOS, iPhone).
///
/// Inserted after PunctuationService.restore() and before
/// VocabularyService.process() in every transcription path.
public enum CapitalizationService {

    // MARK: - Proper noun dictionary

    /// Known proper nouns and their correct casing.
    /// Sorted longest-first so overlapping entries resolve correctly
    /// (e.g. "GitHub" before "Git", "JavaScript" before "Java").
    /// Case-insensitive whole-word matching with regex word boundaries.
    private static let properNouns: [(String, String)] = {
        let entries: [(String, String)] = [
            // Multi-word first
            ("VS Code", "VS Code"),
            // Products — mixed/leading-lowercase casing
            ("ChatGPT", "ChatGPT"),
            ("GitHub", "GitHub"),
            ("OpenAI", "OpenAI"),
            ("Moonshine", "Moonshine"),
            ("JavaScript", "JavaScript"),
            ("TypeScript", "TypeScript"),
            ("iTerm", "iTerm"),
            ("iPhone", "iPhone"),
            ("iPad", "iPad"),
            ("macOS", "macOS"),
            ("iOS", "iOS"),
            ("Xcode", "Xcode"),
            ("Safari", "Safari"),
            ("Chrome", "Chrome"),
            ("Firefox", "Firefox"),
            ("Claude", "Claude"),
            // Companies
            ("Anthropic", "Anthropic"),
            ("Microsoft", "Microsoft"),
            ("Google", "Google"),
            ("Apple", "Apple"),
            ("Slack", "Slack"),
            ("Discord", "Discord"),
            ("Cursor", "Cursor"),
            ("Meta", "Meta"),
            // Protocols/formats
            ("Kubernetes", "Kubernetes"),
            ("Docker", "Docker"),
            ("Python", "Python"),
            ("Linux", "Linux"),
            ("Swift", "Swift"),
            ("Rust", "Rust"),
            ("HTML", "HTML"),
            ("CSS", "CSS"),
            ("SQL", "SQL"),
            ("SSH", "SSH"),
            ("REST", "REST"),
            ("HTTP", "HTTP"),
            ("JSON", "JSON"),
            ("YAML", "YAML"),
            ("API", "API"),
            // Common
            ("Mac", "Mac"),
            ("Git", "Git"),
        ]
        // ponytail: longest-first, then alphabetical for deterministic tiebreak
        return entries.sorted { a, b in
            let lenA = a.0.count, lenB = b.0.count
            if lenA != lenB { return lenA > lenB }
            return a.0 < b.0
        }
    }()

    // MARK: - Public API

    /// Apply all capitalization rules to the given text.
    ///
    /// Idempotent: calling twice on the same input produces the same output.
    /// Empty and punctuation-only strings are returned unchanged.
    /// Grapheme-cluster safe: uses Character-level iteration.
    ///
    /// Order matters:
    /// 1. Sentence-start capitalization — first letter + after . ! ?
    /// 2. Proper noun dictionary — case-insensitive override fixes any
    ///    incorrect casing from step 1 (e.g. "Macos" → "macOS").
    /// 3. Standalone pronoun "i" → "I"
    public static func capitalize(_ text: String) -> String {
        guard !text.isEmpty else { return text }

        var result = text

        // 1. Sentence-start capitalization — first letter + after . ! ?
        result = capitalizeSentences(result)

        // 2. Proper noun dictionary — after sentence-start so mixed-casing
        //    entries (macOS, iOS, iTerm) can override incorrect initial caps.
        result = applyDictionary(result)

        // 3. Standalone pronoun "i" → "I"
        result = capitalizePronounI(result)

        return result
    }

    // MARK: - Private

    /// Replace known proper nouns with their correct casing.
    /// Uses escaped literal matching with word boundaries — no regex
    /// interpolation of unescaped dictionary keys.
    private static func applyDictionary(_ text: String) -> String {
        var result = text
        for (_, proper) in properNouns {
            let escaped = NSRegularExpression.escapedPattern(for: proper)
            let pattern = "\\b\(escaped)\\b"
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
                continue
            }
            let range = NSRange(location: 0, length: (result as NSString).length)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: proper)
        }
        return result
    }

    /// Capitalize the first alphabetic character of the text and the first
    /// alphabetic character after each sentence-ending punctuation mark
    /// (. ! ?). Whitespace, quotes, and other non-letter characters between
    /// the punctuation and the next letter are skipped.
    ///
    /// Character-level iteration preserves grapheme clusters (handles
    /// composed Unicode correctly).
    private static func capitalizeSentences(_ text: String) -> String {
        var result = ""
        var capitalizeNext = true

        for char in text {
            if capitalizeNext && char.isLetter {
                result.append(String(char).uppercased())
                capitalizeNext = false
            } else {
                result.append(char)
                if String(char) == "." || String(char) == "!" || String(char) == "?" {
                    capitalizeNext = true
                }
            }
        }

        return result
    }

    /// Replace standalone "i" (word boundaries both sides) with "I".
    /// Case-insensitive: already-capital "I" is matched and replaced
    /// with "I" — this is what makes the overall capitalize() idempotent.
    private static func capitalizePronounI(_ text: String) -> String {
        guard let regex = try? NSRegularExpression(pattern: "\\bi\\b", options: [.caseInsensitive]) else {
            return text
        }
        let range = NSRange(location: 0, length: (text as NSString).length)
        return regex.stringByReplacingMatches(in: text, range: range, withTemplate: "I")
    }
}
