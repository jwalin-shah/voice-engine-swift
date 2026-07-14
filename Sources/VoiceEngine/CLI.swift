import AppKit
import Foundation

// MARK: - Bench (CLI mode)

public enum Bench {
    public static func bench(file path: String) {
        guard let (samples, duration) = loadWAV(path) else {
            print("Failed to load WAV: \(path)")
            exit(1)
        }
        print("Loaded: \(path)")
        print("  Duration: \(String(format: "%.2f", duration))s, Samples: \(samples.count)")
        let ampMax = samples.prefix(160000).map { abs($0) }.max() ?? 0
        print("  Amplitude: max=\(ampMax)")

        let engine = MoonshineEngine()
        let t0 = CFAbsoluteTimeGetCurrent()
        do {
            try engine.load()
            let loadMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            print("  Model load: \(String(format: "%.0f", loadMs))ms")
        } catch {
            print("Engine load failed: \(error)")
            exit(1)
        }

        let t1 = CFAbsoluteTimeGetCurrent()
        do {
            var timing: TranscribeTiming? = nil
            let text = try engine.transcribe(rawAudio: samples, timing: &timing)
            let elapsed = (CFAbsoluteTimeGetCurrent() - t1) * 1000
            print("  Transcribe: \(String(format: "%.0f", elapsed))ms")
            print("  Text: \"\(text)\"")
        } catch {
            print("Transcription failed: \(error)")
        }
    }

    public static func loadWAV(_ path: String) -> (samples: [Float], duration: Double)? {
        guard let file = FileHandle(forReadingAtPath: path) else { return nil }
        defer { file.closeFile() }
        let data = file.readDataToEndOfFile()
        guard data.count > 12 else { return nil }
        // Verify RIFF + WAVE header
        guard data.prefix(4).elementsEqual("RIFF".utf8),
              data.dropFirst(8).prefix(4).elementsEqual("WAVE".utf8) else { return nil }

        // Scan RIFF chunks to find fmt and data (handles JUNK chunks).
        var offset = 12
        var channels: UInt16 = 1, sampleRate: UInt32 = 16000, bitsPerSample: UInt16 = 16
        var audioData = Data()

        while offset + 8 <= data.count {
            let chunkID = data[offset..<offset+4]
            let chunkSize = data.withUnsafeBytes { $0.load(fromByteOffset: offset + 4, as: UInt32.self) }
            let chunkEnd = offset + 8 + Int(chunkSize)
            guard chunkEnd <= data.count else { break }

            if chunkID.elementsEqual("fmt ".utf8) {
                let fmt = data[offset+8..<chunkEnd]
                guard fmt.count >= 16 else { break }
                channels = fmt.withUnsafeBytes { $0.load(as: UInt16.self) }
                sampleRate = fmt.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) }
                bitsPerSample = fmt.withUnsafeBytes { $0.load(fromByteOffset: 14, as: UInt16.self) }
            } else if chunkID.elementsEqual("data".utf8) {
                audioData = data[offset+8..<chunkEnd]
            }

            offset = chunkEnd
            if offset % 2 != 0 { offset += 1 }  // pad to even boundary
        }

        guard !audioData.isEmpty else { return nil }

        var floats: [Float] = []
        guard bitsPerSample == 16 else { return nil }
        let totalSamples = audioData.count / (Int(channels) * 2)
        floats.reserveCapacity(totalSamples)
        for i in 0..<totalSamples {
            let offset = i * Int(channels) * 2
            let s = audioData.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Int16.self) }
            floats.append(Float(s) / 32768.0)
        }

        let duration = Double(floats.count) / Double(sampleRate)
        if sampleRate != 16000 {
            let ratio = 16000.0 / Double(sampleRate)
            let newCount = Int(Double(floats.count) * ratio)
            var resampled = [Float]()
            resampled.reserveCapacity(newCount)
            for i in 0..<newCount {
                let srcPos = Double(i) / ratio
                let srcIdx = Int(srcPos)
                let frac = srcPos - Double(srcIdx)
                if srcIdx + 1 < floats.count {
                    let s0 = floats[srcIdx], s1 = floats[srcIdx + 1]
                    resampled.append(s0 + Float(frac) * (s1 - s0))
                } else {
                    resampled.append(floats.last ?? 0)
                }
            }
            floats = resampled
        }
        return (floats, duration)
    }
}

@MainActor public final class AppDelegate: NSObject, NSApplicationDelegate {
    private var controller: AppController?
    public func applicationDidFinishLaunching(_ notification: Notification) {
        controller = AppController()
        controller?.run()
    }
    public func applicationWillTerminate(_ notification: Notification) {
        controller = nil
    }
}
