import Foundation

// MARK: - Corpus types

public struct CorpusEntry: Codable {
    public let id: String
    public let wav_path: String
    public let ref_text: String
    public let category: String
    public let audio_secs: Double?

    public init(id: String, wav_path: String, ref_text: String, category: String, audio_secs: Double?) {
        self.id = id
        self.wav_path = wav_path
        self.ref_text = ref_text
        self.category = category
        self.audio_secs = audio_secs
    }
}

// MARK: - Per-utterance result

public struct UtteranceResult: Codable {
    public let id: String
    public let wav_path: String
    public let ref_text: String
    public let hyp_text: String
    public let category: String
    public let wer: Double
    public let cer: Double
    public let deletions: Double
    public let insertions: Double
    public let substitutions: Double
    public let hallucination: Bool
    public let audio_secs: Double
    public let rtf: Double
    public let encoder_ms: Double
    public let cross_kv_ms: Double
    public let decoder_ms: Double
    public let total_ms: Double

    public init(id: String, wav_path: String, ref_text: String, hyp_text: String,
                category: String, wer: Double, cer: Double,
                deletions: Double, insertions: Double, substitutions: Double,
                hallucination: Bool, audio_secs: Double, rtf: Double,
                encoder_ms: Double, cross_kv_ms: Double, decoder_ms: Double, total_ms: Double) {
        self.id = id
        self.wav_path = wav_path
        self.ref_text = ref_text
        self.hyp_text = hyp_text
        self.category = category
        self.wer = wer
        self.cer = cer
        self.deletions = deletions
        self.insertions = insertions
        self.substitutions = substitutions
        self.hallucination = hallucination
        self.audio_secs = audio_secs
        self.rtf = rtf
        self.encoder_ms = encoder_ms
        self.cross_kv_ms = cross_kv_ms
        self.decoder_ms = decoder_ms
        self.total_ms = total_ms
    }
}

// MARK: - Aggregate result

public struct CategoryAggregate: Codable {
    public let category: String
    public let count: Int
    public let avg_wer: Double
    public let avg_cer: Double
    public let avg_rtf: Double
    public let hallucination_rate: Double

    public init(category: String, count: Int, avg_wer: Double, avg_cer: Double,
                avg_rtf: Double, hallucination_rate: Double) {
        self.category = category
        self.count = count
        self.avg_wer = avg_wer
        self.avg_cer = avg_cer
        self.avg_rtf = avg_rtf
        self.hallucination_rate = hallucination_rate
    }
}

public struct AggregateResult: Codable {
    public let model_name: String
    public let total_utterances: Int
    public let avg_wer: Double
    public let avg_cer: Double
    public let avg_rtf: Double
    public let hallucination_rate: Double
    public let deletion_rate: Double
    public let avg_total_ms: Double
    public let by_category: [CategoryAggregate]
}

// MARK: - Metrics calculation

public enum Metrics {
    /// LibriSpeech-style normalization: uppercase, strip non-alpha-non-apostrophe, split on whitespace.
    public static func normalize(_ text: String) -> [String] {
        let upper = text.uppercased()
        // Strip punctuation except apostrophe within words
        let cleaned = upper.replacingOccurrences(
            of: "[^A-Z' ]", with: " ", options: .regularExpression)
        return cleaned.split(separator: " ").map { String($0) }.filter { !$0.isEmpty }
    }

    /// Normalize to character list for CER.
    public static func normalizeChars(_ text: String) -> [Character] {
        let upper = text.uppercased()
        let cleaned = upper.replacingOccurrences(
            of: "[^A-Z' ]", with: "", options: .regularExpression)
        return Array(cleaned).filter { $0 != " " }
    }

    /// Word Error Rate via Levenshtein edit distance on word lists.
    /// Returns (wer, deletions, insertions, substitutions) as fractions of reference length.
    public static func werDetails(ref: [String], hyp: [String]) -> (wer: Double, deletions: Double, insertions: Double, substitutions: Double) {
        let n = ref.count, m = hyp.count
        if n == 0 {
            return (m == 0 ? 0.0 : 1.0, 0.0, Double(m) / Double(max(1, m)), 0.0)
        }
        // Guard: empty hypothesis with non-empty reference = all deletions.
        if m == 0 {
            return (1.0, 1.0, 0.0, 0.0)
        }
        var dp = [[Int]](repeating: [Int](repeating: 0, count: m + 1), count: n + 1)
        for i in 0...n { dp[i][0] = i }
        for j in 0...m { dp[0][j] = j }
        for i in 1...n {
            for j in 1...m {
                dp[i][j] = ref[i-1] == hyp[j-1]
                    ? dp[i-1][j-1]
                    : 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            }
        }
        // Backtrace for detailed error counts
        var i = n, j = m
        var deletions = 0, insertions = 0, substitutions = 0
        while i > 0 || j > 0 {
            if i > 0, j > 0, ref[i-1] == hyp[j-1] {
                i -= 1; j -= 1
            } else if i > 0, j > 0, dp[i][j] == dp[i-1][j-1] + 1 {
                substitutions += 1; i -= 1; j -= 1
            } else if i > 0, dp[i][j] == dp[i-1][j] + 1 {
                deletions += 1; i -= 1
            } else {
                insertions += 1; j -= 1
            }
        }
        let wer = Double(dp[n][m]) / Double(n)
        return (wer, Double(deletions) / Double(n), Double(insertions) / Double(n), Double(substitutions) / Double(n))
    }

    /// Word Error Rate.
    public static func wer(ref: [String], hyp: [String]) -> Double {
        return werDetails(ref: ref, hyp: hyp).wer
    }

    /// Character Error Rate via Levenshtein on character lists.
    public static func cer(ref: [Character], hyp: [Character]) -> Double {
        let n = ref.count, m = hyp.count
        if n == 0 { return m == 0 ? 0.0 : 1.0 }
        if m == 0 { return 1.0 }
        var dp = [[Int]](repeating: [Int](repeating: 0, count: m + 1), count: n + 1)
        for i in 0...n { dp[i][0] = i }
        for j in 0...m { dp[0][j] = j }
        for i in 1...n {
            for j in 1...m {
                dp[i][j] = ref[i-1] == hyp[j-1]
                    ? dp[i-1][j-1]
                    : 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            }
        }
        return Double(dp[n][m]) / Double(n)
    }

    /// Hallucination detection: heuristic based on WER and length ratio.
    /// A hallucination has very high WER (> 0.8) or hypothesis is >3x reference length.
    public static func isHallucination(ref: [String], hyp: [String]) -> Bool {
        let w = wer(ref: ref, hyp: hyp)
        if w > 0.8 { return true }
        if ref.count > 0 && Double(hyp.count) / Double(ref.count) > 3.0 { return true }
        return false
    }
}

// MARK: - Benchmark runner

public enum BenchmarkRunner {
    /// Run benchmark over a corpus, writing per-utterance results as JSONL and returning aggregate.
    public static func run(modelDir: URL?, corpusPath: String, outputDir: String) throws -> AggregateResult {
        // Load corpus (JSONL: one JSON object per line)
        let corpusData = try String(contentsOf: URL(fileURLWithPath: corpusPath), encoding: .utf8)
        let decoder = JSONDecoder()
        let corpus: [CorpusEntry] = try corpusData
            .split(separator: "\n")
            .filter { !$0.isEmpty }
            .map { line in
                guard let data = line.data(using: .utf8) else {
                    throw NSError(domain: "BenchmarkRunner", code: 3,
                                  userInfo: [NSLocalizedDescriptionKey: "Invalid UTF-8 in corpus line"])
                }
                return try decoder.decode(CorpusEntry.self, from: data)
            }
        guard !corpus.isEmpty else {
            throw NSError(domain: "BenchmarkRunner", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Corpus is empty"])
        }

        print("Corpus: \(corpus.count) utterances")
        let engine = MoonshineEngine(modelDir: modelDir)

        // Warmup: transcribe silence to prime CoreML caches.
        // ponytail: removed. The 1s-silence warmup corrupts decoder state for
        // models with different hidden dims (base=416 vs tiny=288). CoreML
        // compilation is fast enough (~1ms) that warmup isn't worth the risk.
        let tLoad = CFAbsoluteTimeGetCurrent()
        try engine.load()
        let loadMs = (CFAbsoluteTimeGetCurrent() - tLoad) * 1000
        print("Model loaded in \(String(format: "%.0f", loadMs))ms")

        // Create output directory
        let outDir = URL(fileURLWithPath: outputDir)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        let modelName = modelDir?.lastPathComponent ?? "moonshine-tiny"
        let resultsPath = outDir.appendingPathComponent("\(modelName).jsonl")
        let aggregatePath = outDir.appendingPathComponent("\(modelName)_aggregate.json")

        // Create empty results file first, then open for writing
        FileManager.default.createFile(atPath: resultsPath.path, contents: nil)
        guard let resultsHandle = FileHandle(forWritingAtPath: resultsPath.path) else {
            throw NSError(domain: "BenchmarkRunner", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot open results file at \(resultsPath.path)"])
        }
        defer { try? resultsHandle.close() }

        var results: [UtteranceResult] = []
        let encoder = JSONEncoder()

        for (idx, entry) in corpus.enumerated() {
            // Skip entries without reference text
            if entry.ref_text.trimmingCharacters(in: .whitespaces).isEmpty {
                print("[\(idx+1)/\(corpus.count)] SKIP \(entry.id) — no reference text")
                continue
            }

            let wavPath = entry.wav_path
            guard let (samples, audioSecs) = Bench.loadAudio(wavPath) else {
                print("[\(idx+1)/\(corpus.count)] SKIP \(entry.id) — failed to load audio")
                continue
            }

            var timing: TranscribeTiming? = TranscribeTiming()
            let hypText: String
            do {
                hypText = try engine.transcribeLong(rawAudio: samples, timing: &timing)
            } catch {
                print("[\(idx+1)/\(corpus.count)] ERROR \(entry.id): \(error)")
                continue
            }

            let t = timing ?? TranscribeTiming()

            // Compute metrics
            let refWords = Metrics.normalize(entry.ref_text)
            let hypWords = Metrics.normalize(hypText)
            let refChars = Metrics.normalizeChars(entry.ref_text)
            let hypChars = Metrics.normalizeChars(hypText)

            let details = Metrics.werDetails(ref: refWords, hyp: hypWords)
            let cer = Metrics.cer(ref: refChars, hyp: hypChars)
            let hall = Metrics.isHallucination(ref: refWords, hyp: hypWords)
            let rtf = audioSecs > 0 ? (t.totalMs / 1000.0) / audioSecs : 0.0

            let result = UtteranceResult(
                id: entry.id,
                wav_path: entry.wav_path,
                ref_text: entry.ref_text,
                hyp_text: hypText,
                category: entry.category,
                wer: details.wer,
                cer: cer,
                deletions: details.deletions,
                insertions: details.insertions,
                substitutions: details.substitutions,
                hallucination: hall,
                audio_secs: audioSecs,
                rtf: rtf,
                encoder_ms: t.encoderMs,
                cross_kv_ms: t.crossKVMs,
                decoder_ms: t.decoderMs,
                total_ms: t.totalMs
            )

            // Write per-utterance result immediately (JSONL)
            var lineData = try encoder.encode(result)
            lineData.append(10) // newline
            try resultsHandle.write(contentsOf: lineData)
            results.append(result)

            let status = hall ? "HALLUC" : String(format: "WER=%.2f", details.wer)
            print("[\(idx+1)/\(corpus.count)] \(entry.id) \(status) RTF=\(String(format: "%.3f", rtf))")
        }

        try resultsHandle.close()

        // Compute aggregates
        let validResults = results.filter { !$0.hallucination }
        let allResults = results

        let avgWER = validResults.isEmpty ? 0.0 : validResults.map(\.wer).reduce(0, +) / Double(validResults.count)
        let avgCER = validResults.isEmpty ? 0.0 : validResults.map(\.cer).reduce(0, +) / Double(validResults.count)
        let avgRTF = validResults.isEmpty ? 0.0 : validResults.map(\.rtf).reduce(0, +) / Double(validResults.count)
        let hallRate = allResults.isEmpty ? 0.0 : Double(allResults.filter(\.hallucination).count) / Double(allResults.count)
        let avgDelRate = validResults.isEmpty ? 0.0 : validResults.map(\.deletions).reduce(0, +) / Double(validResults.count)
        let avgTotalMs = validResults.isEmpty ? 0.0 : validResults.map(\.total_ms).reduce(0, +) / Double(validResults.count)

        // Per-category aggregates
        let grouped = Dictionary(grouping: validResults, by: { $0.category })
        let byCategory: [CategoryAggregate] = grouped.map { (cat, catResults) in
            let count = catResults.count
            let sumWER = catResults.map(\.wer).reduce(0, +)
            let sumCER = catResults.map(\.cer).reduce(0, +)
            let sumRTF = catResults.map(\.rtf).reduce(0, +)
            return CategoryAggregate(
                category: cat,
                count: count,
                avg_wer: sumWER / Double(count),
                avg_cer: sumCER / Double(count),
                avg_rtf: sumRTF / Double(count),
                hallucination_rate: 0.0
            )
        }.sorted { $0.category < $1.category }

        let aggregate = AggregateResult(
            model_name: modelName,
            total_utterances: allResults.count,
            avg_wer: avgWER,
            avg_cer: avgCER,
            avg_rtf: avgRTF,
            hallucination_rate: hallRate,
            deletion_rate: avgDelRate,
            avg_total_ms: avgTotalMs,
            by_category: byCategory
        )

        // Write aggregate JSON
        let aggData = try encoder.encode(aggregate)
        let prettyAgg = try JSONSerialization.jsonObject(with: aggData)
        let prettyData = try JSONSerialization.data(withJSONObject: prettyAgg, options: .prettyPrinted)
        try prettyData.write(to: aggregatePath)

        // Print summary
        print("")
        print("=== Aggregate Results ===")
        print("Model:       \(modelName)")
        print("Utterances:  \(allResults.count)")
        print("Avg WER:     \(String(format: "%.4f", avgWER))")
        print("Avg CER:     \(String(format: "%.4f", avgCER))")
        print("Avg RTF:     \(String(format: "%.4f", avgRTF))")
        print("Halluc rate: \(String(format: "%.1f", hallRate * 100))%")
        print("Deletion:    \(String(format: "%.1f", avgDelRate * 100))%")
        print("Avg total:   \(String(format: "%.1f", avgTotalMs))ms")
        if !byCategory.isEmpty {
            print("--- By Category ---")
            for cat in byCategory {
                print("  \(cat.category): WER=\(String(format: "%.3f", cat.avg_wer)) RTF=\(String(format: "%.3f", cat.avg_rtf)) n=\(cat.count)")
            }
        }
        print("")
        print("Results:     \(resultsPath.path)")
        print("Aggregate:   \(aggregatePath.path)")

        return aggregate
    }
}
