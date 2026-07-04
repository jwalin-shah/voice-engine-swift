import Foundation

/// Simple energy-based voice activity detector.
/// Computes RMS energy over windows and returns true if speech is detected.
public struct VAD {
    /// Minimum RMS energy to consider as speech (tune this based on mic sensitivity)
    public let threshold: Float

    /// Window size in samples at 16kHz (default: 30ms = 480 samples)
    public let windowSize: Int

    /// Minimum fraction of windows that must be above threshold
    public let minActiveRatio: Float

    public init(threshold: Float = 0.001, windowSize: Int = 480, minActiveRatio: Float = 0.1) {
        self.threshold = threshold
        self.windowSize = windowSize
        self.minActiveRatio = minActiveRatio
    }

    /// Returns true if the audio buffer contains speech.
    /// - Parameter samples: float32 audio samples at 16kHz
    public func isSpeech(_ samples: [Float]) -> Bool {
        guard samples.count >= windowSize else {
            // Too short -- compute RMS on the whole buffer
            let rms = sqrt(samples.reduce(0) { $0 + $1 * $1 } / Float(max(samples.count, 1)))
            return rms >= threshold
        }

        var activeWindows: Int = 0
        let totalWindows = samples.count / windowSize

        for i in 0..<totalWindows {
            let start = i * windowSize
            let end = min(start + windowSize, samples.count)
            let windowSamples = samples[start..<end]
            let sumSq = windowSamples.reduce(0) { $0 + $1 * $1 }
            let rms = sqrt(sumSq / Float(windowSamples.count))
            if rms >= threshold {
                activeWindows += 1
            }
        }

        let activeRatio = Float(activeWindows) / Float(max(totalWindows, 1))
        return activeRatio >= minActiveRatio
    }
}
