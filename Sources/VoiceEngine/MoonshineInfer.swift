import Accelerate
import CoreML
import Foundation

// MARK: - Constants

private enum C {
    static let bucketSizes: [Int] = [16000, 48000, 80000, 160000]
    static let S_MAX: Int = 128
    static let S_ENC_MAX: Int = 500
    static let ROT_DIM: Int = 32

    // Chunking for long audio
    static let chunkSamples: Int = 160000        // 10s at 16kHz
    static let overlapSamples: Int = 32000       // 2s overlap
    static let minChunkSamples: Int = 16000      // 1s minimum
}

public enum MoonshineError: LocalizedError {
    case notLoaded, modelNotFound(String), weightsNotFound(String),
         tokenizerNotFound(String), inferenceFailed(String), emptyAudio
    public var errorDescription: String? {
        switch self {
        case .notLoaded: return "Engine not loaded — call load() first"
        case .modelNotFound(let m): return "CoreML model not found: \(m)"
        case .weightsNotFound(let m): return "Weights not found: \(m)"
        case .tokenizerNotFound(let m): return "Tokenizer not found: \(m)"
        case .inferenceFailed(let m): return "Inference failed: \(m)"
        case .emptyAudio: return "Audio buffer is empty"
        }
    }
}

// MARK: - MoonshineEngine

public final class MoonshineEngine: @unchecked Sendable {
    private var encoder: MLModel?
    private var decoder: MLModel?
    private var arch: ArchConfig?
    private var tokenizer: SPModel?
    private var ready = false
    private let modelDir: URL
    private var cosMLArrays: [MLMultiArray]?
    private var sinMLArrays: [MLMultiArray]?

    struct ArchConfig {
        let NL: Int, H: Int, D: Int, HID: Int, S_MAX: Int, S_ENC_MAX: Int
        let ROT_DIM: Int
    }

    public init(modelDir: URL? = nil) {
        self.modelDir = modelDir ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/moonshine-coreml/tiny-streaming")
    }

    // MARK: - Load

    public func load() throws {
        let encDir = modelDir.appendingPathComponent("encoder.mlpackage")
        let decDir = modelDir.appendingPathComponent("decoder_stateful.mlpackage")
        let configPath = modelDir.appendingPathComponent("weights/config.json")
        let spmPath = modelDir.appendingPathComponent("id_to_piece.json")

        guard FileManager.default.fileExists(atPath: encDir.path) else {
            throw MoonshineError.modelNotFound(encDir.path)
        }
        guard FileManager.default.fileExists(atPath: decDir.path) else {
            throw MoonshineError.modelNotFound(decDir.path)
        }
        guard FileManager.default.fileExists(atPath: configPath.path) else {
            throw MoonshineError.weightsNotFound(configPath.path)
        }

        // Compile .mlpackage -> .mlmodelc if needed, then load
        let encCompiled: URL
        if encDir.pathExtension == "mlpackage" {
            encCompiled = try MLModel.compileModel(at: encDir)
        } else {
            encCompiled = encDir
        }
        let encCfg = MLModelConfiguration()
        encCfg.computeUnits = .cpuAndNeuralEngine
        encoder = try MLModel(contentsOf: encCompiled, configuration: encCfg)

        // Load bucket encoders for variable-length audio (1s/3s/5s/10s).
        let decCompiled: URL
        if decDir.pathExtension == "mlpackage" {
            decCompiled = try MLModel.compileModel(at: decDir)
        } else {
            decCompiled = decDir
        }
        let decCfg = MLModelConfiguration()
        decCfg.computeUnits = .cpuOnly
        decoder = try MLModel(contentsOf: decCompiled, configuration: decCfg)

        let cfgData = try Data(contentsOf: configPath)
        let json = try JSONSerialization.jsonObject(with: cfgData) as! [String: Int]
        let a = ArchConfig(
            NL: json["NL"]!, H: json["H"]!, D: json["D"]!, HID: json["HID"]!,
            S_MAX: json["S_MAX"]!, S_ENC_MAX: json["S_ENC_MAX"]!,
            ROT_DIM: json["ROT_DIM"]!
        )
        arch = a

        if FileManager.default.fileExists(atPath: spmPath.path) {
            tokenizer = try SPModel(path: spmPath.path)
        } else {
            throw MoonshineError.tokenizerNotFound(spmPath.path)
        }

        // Pre-compute cos/sin MLMultiArrays for all decoder steps.
        let weightsDir = modelDir.appendingPathComponent("weights")
        let cosTables = try loadF32(weightsDir.appendingPathComponent("cos_tables.f32"),
                                     count: a.S_MAX * a.ROT_DIM)
        let sinTables = try loadF32(weightsDir.appendingPathComponent("sin_tables.f32"),
                                     count: a.S_MAX * a.ROT_DIM)
        let S_MAX = a.S_MAX, ROT_DIM = a.ROT_DIM
        var cosArr = [MLMultiArray]()
        var sinArr = [MLMultiArray]()
        cosArr.reserveCapacity(S_MAX)
        sinArr.reserveCapacity(S_MAX)
        let cosShape: [NSNumber] = [1, 1, 1, NSNumber(value: ROT_DIM)]
        let sinShape: [NSNumber] = [1, 1, 1, NSNumber(value: ROT_DIM)]
        for step in 0..<S_MAX {
            let off = step * ROT_DIM
            cosArr.append(try makeMLArray(Array(cosTables[off..<(off + ROT_DIM)]), shape: cosShape))
            sinArr.append(try makeMLArray(Array(sinTables[off..<(off + ROT_DIM)]), shape: sinShape))
        }
        cosMLArrays = cosArr
        sinMLArrays = sinArr

        ready = true
        Foundation.NSLog("[MoonshineEngine] loaded — encoder.ANE + decoder.CPU + SPM")
    }


    // MARK: - Transcribe (single 10s chunk)

    public func transcribe(rawAudio: [Float]) throws -> String {
        guard ready, let arch, let encoder, let decoder else {
            throw MoonshineError.notLoaded
        }
        guard !rawAudio.isEmpty else { throw MoonshineError.emptyAudio }
        // Sanity check: audio should have reasonable amplitude
        var maxSample: Float = 0
        for s in rawAudio { let a = abs(s); if a > maxSample { maxSample = a } }
        if maxSample < 0.001 {
            Foundation.NSLog("[MoonshineEngine] WARNING: near-silent audio (max=\(maxSample))")
        }

        // 1. Pad to nearest bucket (minimizes padding noise in streaming).
        var audio = rawAudio
        let maxSamples = 160000
        if audio.count < maxSamples {
            audio.append(contentsOf: [Float](repeating: 0, count: maxSamples - audio.count))
        } else if audio.count > maxSamples {
            audio = Array(audio[..<maxSamples])
        }

        // 2. Encoder forward (ANE).
        let audioML = try makeMLArray(audio, shape: [1, NSNumber(value: maxSamples)])
        let encInName = encoder.modelDescription.inputDescriptionsByName.first!.key
        let encOutName = encoder.modelDescription.outputDescriptionsByName.first!.key
        let encOut = try encoder.prediction(from: try MLDictionaryFeatureProvider(dictionary: [
            encInName: audioML
        ])).featureValue(for: encOutName)!.multiArrayValue!

        let S_enc = encOut.shape[1].intValue
        let hidden = mlToFloats(encOut)
        // Debug: log first few hidden state values


        // 3. Build cross-KV via Accelerate gemm.
        let weightsDir = modelDir.appendingPathComponent("weights")
        let (crossK, crossV, crossMask) = try buildCrossKV(
            hidden: hidden, S_enc: S_enc, arch: arch, weightsDir: weightsDir)


        // 4. Decoder loop.
        return try decodeLoop(crossK: crossK, crossV: crossV, crossMask: crossMask,
                               S_enc: S_enc, arch: arch, decoder: decoder,
                               weightsDir: weightsDir)
    }

    // MARK: - TranscribeLong (handles arbitrary length via chunking)

    public func transcribeLong(rawAudio: [Float]) throws -> String {
        guard !rawAudio.isEmpty else { throw MoonshineError.emptyAudio }

        // If short enough, do single pass
        if rawAudio.count <= C.chunkSamples {
            return try transcribe(rawAudio: rawAudio)
        }

        let step = C.chunkSamples - C.overlapSamples
        var fullText = ""
        var prevText = ""

        var chunkStart = 0
        while chunkStart < rawAudio.count {
            let chunkEnd = min(chunkStart + C.chunkSamples, rawAudio.count)
            let chunk = Array(rawAudio[chunkStart..<chunkEnd])
            if chunk.count < C.minChunkSamples { break }

            let text = try transcribe(rawAudio: chunk)

            // Dedup overlap with previous chunk
            if !prevText.isEmpty {
                let newPart = dedupOverlap(prevText: prevText, newText: text)
                fullText += " " + newPart
            } else {
                fullText += text
            }

            prevText = text
            chunkStart += step
        }

        return fullText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Overlap dedup (sentence-level)

    private func dedupOverlap(prevText: String, newText: String) -> String {
        let prevSents = prevText.components(separatedBy: CharacterSet(charactersIn: ".!?"))
        let newSents = newText.components(separatedBy: CharacterSet(charactersIn: ".!?"))

        guard !prevSents.isEmpty, !newSents.isEmpty else { return newText }

        // Get last 1-2 sentences of previous chunk
        let tail = prevSents.suffix(2).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }

        // Drop leading sentences from new that overlap with tail
        for skip in 0..<min(newSents.count, 4) {
            let h = newSents[skip].trimmingCharacters(in: .whitespaces).lowercased()
            if h.isEmpty { continue }

            for t in tail {
                let tNorm = t.trimmingCharacters(in: .whitespaces).lowercased()
                if tNorm.isEmpty { continue }
                if h.hasPrefix(tNorm) || tNorm.hasPrefix(h)
                    || (tNorm.count > 15 && h.count > 15
                        && (tNorm.prefix(15) == h.prefix(15)
                            || tNorm.contains(h) || h.contains(tNorm))) {
                    let remaining = newSents.dropFirst(skip + 1).joined(separator: ".")
                    return remaining.trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
            if newSents[skip].split(separator: " ").count <= 4 { continue }
            break
        }

        return newText
    }

    // MARK: - Cross-KV construction

    private func buildCrossKV(hidden: [Float], S_enc: Int, arch: ArchConfig,
                               weightsDir: URL) throws -> (MLMultiArray, MLMultiArray, MLMultiArray) {
        let NL = arch.NL, H = arch.H, D = arch.D, HID = arch.HID
        let HD = H * D
        let S_ENC_MAX = arch.S_ENC_MAX

        let crossK = try MLMultiArray(
            shape: [NSNumber(value: NL), 1, NSNumber(value: H),
                    NSNumber(value: S_ENC_MAX), NSNumber(value: D)],
            dataType: .float32)
        let crossV = try MLMultiArray(
            shape: [NSNumber(value: NL), 1, NSNumber(value: H),
                    NSNumber(value: S_ENC_MAX), NSNumber(value: D)],
            dataType: .float32)
        let crossMask = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: S_ENC_MAX)], dataType: .float32)

        // Bind pointers once for direct memory access (avoids ObjC NSNumber overhead).
        let ckBase = crossK.dataPointer.bindMemory(to: Float.self, capacity: NL * H * S_ENC_MAX * D)
        let cvBase = crossV.dataPointer.bindMemory(to: Float.self, capacity: NL * H * S_ENC_MAX * D)
        let mPtr = crossMask.dataPointer.bindMemory(to: Float.self, capacity: S_ENC_MAX)
        let nBytes = D * MemoryLayout<Float>.stride

        for i in 0..<NL {
            let kw = try loadF32(weightsDir.appendingPathComponent("layer\(i)_k_weight.f32"),
                                 count: HD * HID)
            let vw = try loadF32(weightsDir.appendingPathComponent("layer\(i)_v_weight.f32"),
                                 count: HD * HID)

            var k = gemm(M: S_enc, N: HD, K: HID, A: hidden, B: kw)
            var v = gemm(M: S_enc, N: HD, K: HID, A: hidden, B: vw)

            let kbPath = weightsDir.appendingPathComponent("layer\(i)_k_bias.f32")
            if FileManager.default.fileExists(atPath: kbPath.path) {
                let kb = try loadF32(kbPath, count: HD)
                for j in 0..<k.count { k[j] += kb[j % HD] }
            }
            let vbPath = weightsDir.appendingPathComponent("layer\(i)_v_bias.f32")
            if FileManager.default.fileExists(atPath: vbPath.path) {
                let vb = try loadF32(vbPath, count: HD)
                for j in 0..<v.count { v[j] += vb[j % HD] }
            }

            // Vectorized transpose: (S_enc, H, D) -> (H, S_enc, D) via memcpy per row.
            let layerOff = i * H * S_ENC_MAX * D
            k.withUnsafeMutableBufferPointer { kBuf in
                v.withUnsafeMutableBufferPointer { vBuf in
                    for h in 0..<H {
                        let rowOff = layerOff + h * S_ENC_MAX * D
                        for s in 0..<S_enc {
                            let srcOff = s * HD + h * D
                            let dstOff = rowOff + s * D
                            memcpy(ckBase + dstOff, kBuf.baseAddress! + srcOff, nBytes)
                            memcpy(cvBase + dstOff, vBuf.baseAddress! + srcOff, nBytes)
                        }
                    }
                }
            }
        }

        for s in 0..<S_ENC_MAX {
            mPtr[s] = s < S_enc ? 0.0 : -10000.0
        }

        return (crossK, crossV, crossMask)
    }

    // MARK: - Decoder autoregressive loop

    private func decodeLoop(crossK: MLMultiArray, crossV: MLMultiArray,
                             crossMask: MLMultiArray, S_enc: Int,
                             arch: ArchConfig, decoder: MLModel,
                             weightsDir: URL) throws -> String {
        let S_MAX = arch.S_MAX

        guard let cosMLArrays, let sinMLArrays else {
            throw MoonshineError.notLoaded
        }

        let state = decoder.makeState()

        let BOS: Int32 = 1
        let EOS: Int32 = 2
        var tokenIDs: [Int32] = [BOS]

        // Pre-allocate mutable buffers (avoids ObjC allocation per step).
        let inputIds = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let onehot = try MLMultiArray(
            shape: [1, 1, NSNumber(value: S_MAX), 1], dataType: .float32)
        let onehotPtr = onehot.dataPointer.bindMemory(to: Float.self, capacity: S_MAX)
        let attnMask = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: S_MAX)], dataType: .float32)
        let attnPtr = attnMask.dataPointer.bindMemory(to: Float.self, capacity: S_MAX)
        // Initialize first step: position 0 unlocked.
        onehotPtr[0] = 1.0
        for i in 0..<S_MAX { attnPtr[i] = -10000.0 }
        attnPtr[0] = 0.0

        for step in 0..<min(S_MAX, 200) {
            inputIds[0] = NSNumber(value: tokenIDs.last!)

            let cosML = cosMLArrays[step]
            let sinML = sinMLArrays[step]

            // Update onehot: clear previous, set current.
            if step > 0 { onehotPtr[step - 1] = 0.0 }
            onehotPtr[step] = 1.0

            // Update attn_mask: unlock current position.
            if step > 0 { attnPtr[step] = 0.0 }

            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": inputIds,
                "attn_mask": attnMask,
                "cos": cosML,
                "sin": sinML,
                "write_onehot": onehot,
                "cross_k": crossK,
                "cross_v": crossV,
                "cross_mask": crossMask,
            ])

            let pred = try decoder.prediction(from: input, using: state)

            guard let logitsML = pred.featureValue(for: "logits")?.multiArrayValue else {
                throw MoonshineError.inferenceFailed("nil logits at step \(step)")
            }

            let vocabSize = logitsML.shape[2].intValue
            let ptr = logitsML.dataPointer.bindMemory(to: Float.self, capacity: vocabSize)
            var maxIdx: Int32 = 0
            var maxVal: Float = -.infinity
            for v in 0..<vocabSize {
                let val = ptr[v]
                if val > maxVal { maxVal = val; maxIdx = Int32(v) }
            }
            tokenIDs.append(maxIdx)
            if maxIdx == EOS { break }


        }

        return tokenizer?.decode(tokenIDs) ?? tokenIDs.map { "\($0)" }.joined(separator: " ")
    }


    // MARK: - Tokenizer (JSON-based, no Python)

    private final class SPModel {
        private var idToPiece: [Int32: String] = [:]

        init(path: String) throws {
            let jsonPath = path.replacingOccurrences(
                of: "sentencepiece.bpe.model", with: "id_to_piece.json")
            let data = try Data(contentsOf: URL(fileURLWithPath: jsonPath))
            let json = try JSONSerialization.jsonObject(with: data) as! [String: [String: String]]
            let dict = json["id_to_piece"] ?? [:]
            for (key, value) in dict {
                if let id = Int32(key) {
                    idToPiece[id] = value
                }
            }
        }

        func decode(_ ids: [Int32]) -> String {
            var result = ""
            for id in ids {
                guard id != 1, id != 2 else { continue }
                if let piece = idToPiece[id] { result += piece }
            }
            return result.replacingOccurrences(of: "  ", with: " ")
                         .trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

}

// MARK: - SentencePiece

// MARK: - Math helpers

private func loadF32(_ url: URL, count: Int) throws -> [Float] {
    let data = try Data(contentsOf: url)
    return data.withUnsafeBytes { ptr in
        Array(ptr.bindMemory(to: Float.self).prefix(count))
    }
}

private func gemm(M: Int, N: Int, K: Int, A: [Float], B: [Float]) -> [Float] {
    var C = [Float](repeating: 0, count: M * N)
    A.withUnsafeBufferPointer { aPtr in
        B.withUnsafeBufferPointer { bPtr in
            C.withUnsafeMutableBufferPointer { cPtr in
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            Int(M), Int(N), Int(K),
                            1.0, aPtr.baseAddress!, Int(K),
                            bPtr.baseAddress!, Int(K),
                            0.0, cPtr.baseAddress!, Int(N))
            }
        }
    }
    return C
}

private func mlToFloats(_ ml: MLMultiArray) -> [Float] {
    let ptr = ml.dataPointer.bindMemory(to: Float.self, capacity: ml.count)
    return (0..<ml.count).map { ptr[$0] }
}

private func makeMLArray(_ data: [Float], shape: [NSNumber]) throws -> MLMultiArray {
    let ml = try MLMultiArray(shape: shape, dataType: .float32)
    let ptr = ml.dataPointer.bindMemory(to: Float.self, capacity: data.count)
    ptr.initialize(from: data, count: data.count)
    return ml
}
