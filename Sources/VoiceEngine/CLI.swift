import AppKit
import AVFoundation
import Foundation

// MARK: - Bench (CLI mode)

public enum Bench {
    public static func bench(file path: String, modelDir: String? = nil) {
        guard let (samples, duration) = loadAudio(path) else {
            print("Failed to load audio: \(path)")
            exit(1)
        }
        print("File: \(path)")
        print("  Duration: \(String(format: "%.2f", duration))s, Samples: \(samples.count)")
        let ampMax = samples.prefix(min(160000, samples.count)).map { abs($0) }.max() ?? 0
        print("  Amplitude: max=\(String(format: "%.4f", ampMax))")

        let md = modelDir.map { URL(fileURLWithPath: $0) }
        let engine = MoonshineEngine(modelDir: md)
        var timing: TranscribeTiming? = TranscribeTiming()

        let tLoad = CFAbsoluteTimeGetCurrent()
        do {
            try engine.load()
            let loadMs = (CFAbsoluteTimeGetCurrent() - tLoad) * 1000
            print("  Model load: \(String(format: "%.1f", loadMs))ms")
        } catch {
            print("Engine load failed: \(error)")
            exit(1)
        }

        do {
            let text = try engine.transcribe(rawAudio: samples, timing: &timing)
            if let t = timing {
                print("  Encoder (ANE):     \(String(format: "%5.1f", t.encoderMs))ms")
                print("  Cross-KV (CPU):    \(String(format: "%5.1f", t.crossKVMs))ms")
                print("  Decoder (CPU):     \(String(format: "%5.1f", t.decoderMs))ms")
                print("  Total model:       \(String(format: "%5.1f", t.totalMs))ms")
            }
            if !text.isEmpty {
                let preview = text.count > 80 ? String(text.prefix(80)) + "..." : text
                print("  Transcript: \"\(preview)\"")
            }
        } catch {
            print("Transcription failed: \(error)")
        }
    }

    /// Run benchmark over a corpus JSONL file, outputting per-utterance and aggregate results.
    public static func benchCorpus(modelDir: String?, corpus path: String, outputDir: String) {
        let md = modelDir.map { URL(fileURLWithPath: $0) }
        do {
            _ = try BenchmarkRunner.run(modelDir: md, corpusPath: path, outputDir: outputDir)
        } catch {
            print("Benchmark failed: \(error)")
            exit(1)
        }
    }

    /// Load audio via AVFoundation — handles WAV, AIFF, MP3, M4A, FLAC, etc.
    /// Converts to mono 16 kHz Float32.
    public static func loadAudio(_ path: String) -> (samples: [Float], duration: Double)? {
        let url = URL(fileURLWithPath: path)
        guard let file = try? AVAudioFile(forReading: url) else { return nil }

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        ) else {
            print("Failed to create 16kHz float32 audio format")
            return nil
        }

        let sourceFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard let sourceBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frameCount) else {
            return nil
        }
        do {
            try file.read(into: sourceBuffer)
        } catch {
            print("Failed to read audio file: \(error)")
            return nil
        }

        let channelData: UnsafeMutablePointer<Float>
        let frameLength: AVAudioFrameCount

        if sourceFormat.isEqual(targetFormat) {
            channelData = sourceBuffer.floatChannelData![0]
            frameLength = sourceBuffer.frameLength
        } else {
            guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
                return nil
            }
            let maxOutputFrames = AVAudioFrameCount(Double(frameCount) * targetFormat.sampleRate / sourceFormat.sampleRate + 1024)
            guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: maxOutputFrames) else {
                return nil
            }

            var sourceEof = false
            let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                if sourceEof {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                sourceEof = true
                outStatus.pointee = .haveData
                return sourceBuffer
            }

            var error: NSError?
            converter.convert(to: convertedBuffer, error: &error, withInputFrom: inputBlock)
            if let error {
                print("Audio conversion error: \(error)")
                return nil
            }
            channelData = convertedBuffer.floatChannelData![0]
            frameLength = convertedBuffer.frameLength
        }

        let samples = Array(UnsafeBufferPointer(start: channelData, count: Int(frameLength)))
        let duration = Double(samples.count) / 16000.0
        return (samples, duration)
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
