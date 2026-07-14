import CoreML
import Foundation

/// Punctuation restoration using FullStop (XLM-RoBERTa token-classifier, CoreML).
///
/// Deterministic single forward pass — no autoregressive generation, no hallucination risk.
/// Labels each token with O/./,/?:/- and reconstructs punctuated text.
///
/// Invariants:
/// - Input words are preserved in output order unchanged (no rewriting).
/// - Only allowed punctuation characters are inserted: . , ? : -
/// - Empty input returns empty output.
/// - Inputs longer than maxSeqLen (256 tokens) are truncated.
/// - Output is deterministic for a fixed model and runtime.
public actor PunctuationService {
    /// Punctuation mode — for now just enabled/disabled.
    public enum Mode: String, Codable, CaseIterable {
        case disabled = "Disabled"
        case enabled = "Enabled"
    }

    // MARK: - Labels

    /// Label mapping: model output class → punctuation character.
    /// 0 = no punctuation (O), 1 = period, 2 = comma, 3 = question, 4 = hyphen, 5 = colon.
    private static let idToLabel: [Int: String] = [
        0: "", 1: ".", 2: ",", 3: "?", 4: "-", 5: ":"
    ]

    // MARK: - State

    private var model: MLModel?
    private var tokenizer: FullStopTokenizer?
    private var ready = false

    private let modelDir: URL
    private let compiledCacheDir: URL
    private let maxSeqLen = 256

    public nonisolated var mode: Mode {
        get {
            guard let raw = UserDefaults.standard.string(forKey: "punctuationMode"),
                  let mode = Mode(rawValue: raw) else { return .enabled }
            return mode
        }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: "punctuationMode") }
    }

    // MARK: - Init

    public init(modelDir: URL? = nil) {
        let md = modelDir ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/fullstop-coreml/large")
        self.modelDir = md
        self.compiledCacheDir = md.appendingPathComponent("compiled")
        try? FileManager.default.createDirectory(at: compiledCacheDir, withIntermediateDirectories: true)
    }

    // MARK: - Load

    public func load() throws {
        let mlpackage = modelDir.appendingPathComponent("fullstop-punctuation.mlpackage")

        guard FileManager.default.fileExists(atPath: mlpackage.path) else {
            throw ServiceError.modelNotFound(mlpackage.path)
        }

        // Compile mlpackage → mlmodelc (persistent cache) and load.
        let compiled = try compileOrCached(mlpackage)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: compiled, configuration: config)

        // Load tokenizer.
        tokenizer = try FullStopTokenizer(vocabDir: modelDir)

        ready = true
        Foundation.NSLog("[PunctuationService] loaded — FullStop-large CoreML + Unigram tokenizer")
    }

    /// Always available once loaded — no network, no daemon.
    public func isReady() -> Bool { ready }

    // MARK: - Restore

    /// Restore punctuation in transcribed text.
    ///
    /// - Parameter text: raw lowercase text from ASR output (may contain CleanupService filler-removal artifacts).
    /// - Returns: punctuated text, or original text if disabled or empty.
    public func restore(_ text: String) throws -> String {
        guard ready, let model, let tokenizer else {
            throw ServiceError.notLoaded
        }
        guard mode != .disabled else { return text }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return text }

        // 1. Tokenize.
        let tokResult = tokenizer.encode(trimmed, maxLength: maxSeqLen)

        // 2. CoreML inference.
        let logits = try predict(inputIds: tokResult.inputIds, attentionMask: tokResult.attentionMask, model: model)

        // 3. Argmax → label IDs.
        let labelIds = argmax(logits: logits, seqLen: tokResult.inputIds.count, numLabels: 6)

        // 4. Reconstruct punctuated text.
        let result = reconstruct(
            inputIds: tokResult.inputIds,
            labelIds: labelIds,
            tokenizer: tokenizer
        )

        return result
    }

    // MARK: - Private: CoreML

    private func predict(inputIds: [Int], attentionMask: [Int], model: MLModel) throws -> [Float] {
        // Build MLMultiArray inputs.
        let seqLen = inputIds.count
        let idsML = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
        let maskML = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)

        let idsPtr = idsML.dataPointer.bindMemory(to: Int32.self, capacity: seqLen)
        let maskPtr = maskML.dataPointer.bindMemory(to: Int32.self, capacity: seqLen)

        for i in 0..<seqLen {
            idsPtr[i] = Int32(inputIds[i])
            maskPtr[i] = Int32(attentionMask[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": idsML,
            "attention_mask": maskML,
        ])

        let output = try model.prediction(from: input)

        // Output: logits — shape [1, seq_len, 6]
        guard let logitsML = output.featureValue(for: "logits")?.multiArrayValue else {
            throw ServiceError.inferenceFailed("nil logits output")
        }

        return mlToFloats(logitsML)
    }

    // MARK: - Private: Argmax

    private func argmax(logits: [Float], seqLen: Int, numLabels: Int) -> [Int] {
        var labels: [Int] = []
        labels.reserveCapacity(seqLen)
        for pos in 0..<seqLen {
            let offset = pos * numLabels
            var maxIdx = 0
            var maxVal: Float = -.infinity
            for l in 0..<numLabels {
                let val = logits[offset + l]
                if val > maxVal { maxVal = val; maxIdx = l }
            }
            labels.append(maxIdx)
        }
        return labels
    }

    // MARK: - Private: Reconstruction

    /// Reconstruct punctuated text using word-level label aggregation.
    ///
    /// For each word (tokens starting with ▁ + following subwords),
    /// use the label of the FIRST subword token. Append punctuation after
    /// the complete word. This matches the official
    /// ``deepmultilingualpunctuation`` package behavior.
    private func reconstruct(inputIds: [Int], labelIds: [Int], tokenizer: FullStopTokenizer) -> String {
        var result = ""
        var currentWord = ""
        var wordLabel: String = ""  // punctuation for current word

        for (tid, labelId) in zip(inputIds, labelIds) {
            // Skip special tokens.
            guard tid != tokenizer.bosId,
                  tid != tokenizer.eosId,
                  tid != tokenizer.padId else { continue }

            let labelChar = Self.idToLabel[labelId] ?? ""

            guard let piece = tokenizer.idToPieceString(tid) else { continue }

            if piece.first == FullStopTokenizer.wordBoundary {
                // Flush previous word.
                if !currentWord.isEmpty {
                    result += currentWord + wordLabel + " "
                }
                currentWord = String(piece.dropFirst())  // strip ▁
                wordLabel = labelChar
            } else {
                // Continuation subword — append, keep first subword's label.
                currentWord += piece
            }
        }

        // Flush final word.
        if !currentWord.isEmpty {
            result += currentWord + wordLabel
        }

        // Clean up: collapse multiple spaces, trim.
        var cleaned = result
            .replacingOccurrences(of: " +", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Remove trailing spaces before punctuation that got double-spaced.
        // e.g. "hello ." → "hello."
        cleaned = cleaned.replacingOccurrences(of: " .", with: ".")
        cleaned = cleaned.replacingOccurrences(of: " ,", with: ",")
        cleaned = cleaned.replacingOccurrences(of: " ?", with: "?")
        cleaned = cleaned.replacingOccurrences(of: " -", with: "-")
        cleaned = cleaned.replacingOccurrences(of: " :", with: ":")

        return cleaned
    }

    // MARK: - Private: Model compilation

    private func compileOrCached(_ mlpackage: URL) throws -> URL {
        let name = mlpackage.deletingPathExtension().lastPathComponent
        let cached = compiledCacheDir.appendingPathComponent("\(name).mlmodelc")

        let srcDate = (try? FileManager.default.attributesOfItem(atPath: mlpackage.path)[.modificationDate] as? Date) ?? .distantPast
        if let cachedDate = try? FileManager.default.attributesOfItem(atPath: cached.path)[.modificationDate] as? Date,
           cachedDate >= srcDate {
            return cached
        }

        let compiled = try MLModel.compileModel(at: mlpackage)
        try? FileManager.default.removeItem(at: cached)
        try FileManager.default.copyItem(at: compiled, to: cached)
        return cached
    }

    // MARK: - Errors

    public enum ServiceError: LocalizedError {
        case notLoaded
        case modelNotFound(String)
        case inferenceFailed(String)

        public var errorDescription: String? {
            switch self {
            case .notLoaded: return "PunctuationService not loaded — call load() first"
            case .modelNotFound(let path): return "FullStop model not found: \(path)"
            case .inferenceFailed(let msg): return "Inference failed: \(msg)"
            }
        }
    }
}

// MARK: - Math helpers

private func mlToFloats(_ ml: MLMultiArray) -> [Float] {
    let ptr = ml.dataPointer.bindMemory(to: Float.self, capacity: ml.count)
    return (0..<ml.count).map { ptr[$0] }
}
