import Foundation

/// XLM-RoBERTa Unigram SentencePiece tokenizer for FullStop punctuation model.
///
/// Uses a trie-based prefix lookup + Viterbi decoding for tokenization.
/// Loads vocabulary from ``id_to_piece.json`` format (same as Moonshine)
/// and a reverse lookup dict from ``tokenizer_compact.json``.
public final class FullStopTokenizer: Sendable {
    /// Piece string → token ID (for encoding).
    private let pieceToId: [String: Int]
    /// Token ID → piece string (for decoding).
    private let idToPiece: [Int: String]
    /// Trie root for efficient prefix matching during Viterbi encoding.
    private let trieRoot: TrieNode

    // Special token IDs.
    public let bosId: Int = 0
    public let padId: Int = 1
    public let eosId: Int = 2
    public let unkId: Int = 3

    /// Word boundary marker (U+2581).
    public static let wordBoundary: Character = "\u{2581}"

    // MARK: - Trie

    /// Mutable trie node — used only during init; safe because init is single-threaded.
    private final class TrieNode: @unchecked Sendable {
        var children: [Character: TrieNode] = [:]
        var tokenId: Int?  // non-nil if this node represents a complete token
        var score: Float = -.infinity  // log probability of this piece
    }

    // MARK: - Init

    /// Load tokenizer from the exported vocabulary files.
    /// - Parameters:
    ///   - vocabDir: directory containing `tokenizer_compact.json` or `id_to_piece.json`
    public init(vocabDir: URL) throws {
        let compactPath = vocabDir.appendingPathComponent("tokenizer_compact.json")

        guard FileManager.default.fileExists(atPath: compactPath.path) else {
            throw TokenizerError.fileNotFound(compactPath.path)
        }

        let data = try Data(contentsOf: compactPath)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        // Parse vocab: list of [piece, score] pairs
        guard let vocabList = json["vocab"] as? [[Any]] else {
            throw TokenizerError.invalidFormat("vocab is not a list")
        }

        var p2i: [String: Int] = [:]
        var i2p: [Int: String] = [:]
        let root = TrieNode()

        for (idx, entry) in vocabList.enumerated() {
            guard entry.count >= 2,
                  let piece = entry[0] as? String,
                  let scoreNum = entry[1] as? NSNumber else {
                throw TokenizerError.invalidFormat("vocab[\(idx)] has wrong shape")
            }
            let score = scoreNum.floatValue

            p2i[piece] = idx
            i2p[idx] = piece

            // Insert into trie (skip special tokens for encoding)
            if idx >= 4 {  // skip <s>, <pad>, </s>, <unk>
                var node = root
                for ch in piece {
                    if node.children[ch] == nil {
                        node.children[ch] = TrieNode()
                    }
                    node = node.children[ch]!
                }
                node.tokenId = idx
                node.score = score
            }
        }

        // Add special token entries for completeness
        i2p[0] = "<s>"
        i2p[1] = "<pad>"
        i2p[2] = "</s>"
        i2p[3] = "<unk>"

        self.pieceToId = p2i
        self.idToPiece = i2p
        self.trieRoot = root
    }

    // MARK: - Encode

    /// Tokenize text into token IDs with BOS/EOS.
    /// - Parameter text: raw text to tokenize
    /// - Parameter maxLength: maximum sequence length (including BOS/EOS)
    /// - Returns: ``TokenizeResult`` with input IDs and attention mask.
    public func encode(_ text: String, maxLength: Int) -> TokenizeResult {
        let normalized = preTokenize(text)
        var ids: [Int] = [bosId]

        for segment in normalized {
            let word = String(segment)
            let encoded = encodeWord(word)
            ids.append(contentsOf: encoded)
            if ids.count >= maxLength - 1 { break }  // reserve space for EOS
        }

        ids.append(eosId)

        // Pad to maxLength
        let seqLen = min(ids.count, maxLength)
        let paddedIds = Array(ids.prefix(maxLength))
        var attentionMask = [Int](repeating: 1, count: seqLen)
        if paddedIds.count > seqLen {
            attentionMask.append(contentsOf: [Int](repeating: 0, count: paddedIds.count - seqLen))
        }
        // Pad with PAD token
        let finalIds: [Int]
        if paddedIds.count < maxLength {
            finalIds = paddedIds + [Int](repeating: padId, count: maxLength - paddedIds.count)
            var mask = [Int](repeating: 1, count: seqLen)
            mask.append(contentsOf: [Int](repeating: 0, count: maxLength - seqLen))
            return TokenizeResult(inputIds: finalIds, attentionMask: mask)
        } else {
            return TokenizeResult(inputIds: paddedIds, attentionMask: attentionMask)
        }
    }

    /// TokenizeResult with input IDs and attention mask for CoreML inference.
    public struct TokenizeResult {
        public let inputIds: [Int]
        public let attentionMask: [Int]
    }

    // MARK: - Decode

    /// Decode token IDs back to text (for reconstruction).
    public func decode(_ ids: [Int]) -> String {
        var result = ""
        for id in ids {
            guard id != bosId, id != eosId, id != padId else { continue }
            if let piece = idToPiece[id] {
                result += piece
            }
        }
        // Replace word boundary markers with spaces.
        return result
            .replacingOccurrences(of: String(Self.wordBoundary), with: " ")
            .replacingOccurrences(of: " +", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Decode a single token ID to its piece string.
    public func idToPieceString(_ id: Int) -> String? {
        idToPiece[id]
    }

    // MARK: - Private: Encoding

    /// Pre-tokenize: split text into segments using XLM-RoBERTa's regex.
    /// The HF tokenizer uses a regex that splits on whitespace while preserving
    /// certain patterns. We approximate this with a simpler split.
    private func preTokenize(_ text: String) -> [Substring] {
        // XLM-RoBERTa regex splits text into tokens at whitespace boundaries,
        // keeping whitespace as separate tokens. For ASR output (lowercase, no
        // special chars), simple whitespace splitting is sufficient.
        guard !text.isEmpty else { return [] }

        // Split on whitespace, keeping each word
        let words = text.split(separator: " ")

        // Add ▁ prefix to each word (XLM-RoBERTa convention).
        // The FIRST word gets ▁ too (BOS is separate).
        return words.map { word in
            Substring(String(Self.wordBoundary) + word)
        }
    }

    /// Encode a single word (already ▁-prefixed) using Viterbi Unigram tokenization.
    private func encodeWord(_ word: String) -> [Int] {
        let chars = Array(word)
        let n = chars.count

        // dp[i] = (best_score, back_pointer) for prefix of length i
        var dpScore = [Float](repeating: -.infinity, count: n + 1)
        var dpBack = [Int](repeating: -1, count: n + 1)
        var dpTokenId = [Int](repeating: -1, count: n + 1)
        dpScore[0] = 0

        for i in 0..<n {
            guard dpScore[i] > -.infinity else { continue }

            var node = trieRoot
            var j = i
            while j < n {
                guard let child = node.children[chars[j]] else { break }
                node = child
                j += 1

                if let tokenId = node.tokenId {
                    let newScore = dpScore[i] + node.score
                    if newScore > dpScore[j] {
                        dpScore[j] = newScore
                        dpBack[j] = i
                        dpTokenId[j] = tokenId
                    }
                }
            }
        }

        // Reconstruct token IDs from back pointers.
        var tokens: [Int] = []
        var pos = n
        while pos > 0 {
            let prev = dpBack[pos]
            guard prev >= 0, dpTokenId[pos] >= 0 else {
                // Fallback: no valid segmentation found, use UNK
                tokens = [unkId]
                break
            }
            tokens.append(dpTokenId[pos])
            pos = prev
        }

        return tokens.reversed()
    }

    // MARK: - Errors

    public enum TokenizerError: LocalizedError {
        case fileNotFound(String)
        case invalidFormat(String)

        public var errorDescription: String? {
            switch self {
            case .fileNotFound(let path): return "Tokenizer file not found: \(path)"
            case .invalidFormat(let msg): return "Invalid tokenizer format: \(msg)"
            }
        }
    }
}
