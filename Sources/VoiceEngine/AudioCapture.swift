import AVFoundation
import CoreAudio

/// Captures mono 16 kHz float32 audio from the default input device on
/// key-down, accumulates into a ring-like buffer, and delivers a single
/// contiguous buffer on stop.
///
/// Uses 64-frame buffers for minimum touch-to-audio latency (1.3 ms at
/// 48 kHz). Device format is left at its native sample rate; the tap
/// delivers whatever the hardware produces. Conversion to 16 kHz happens
/// inline via AVAudioConverter so the Moonshine encoder receives the
/// expected sample rate.
public final class AudioCapture {
    private var engine = AVAudioEngine()
    private var accumulator = Data()
    private var partialAccumulator = Data()
    private let queue = DispatchQueue(label: "voice.audio", qos: .userInitiated)
    private(set) public var isRecording = false
    private let outputFormat: AVAudioFormat = {
        AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    }()
    private var converter: AVAudioConverter?
    public var partialCallback: ((Data) -> Void)?
    private let partialIntervalSamples: Int = 48000  // ~3s at 16kHz — 30% real audio in 10s window — need enough audio to fill the 10s encoder window
    private var lastPartialSampleCount: Int = 0


    /// When true, routes audio through the Mac's built-in mic regardless of
    /// the system default input device.
    public var forceBuiltInMic: Bool = UserDefaults.standard.bool(forKey: "forceBuiltInMic") {
        didSet { UserDefaults.standard.set(forceBuiltInMic, forKey: "forceBuiltInMic") }
    }

    /// The sample rate of the converted output audio (always 16000).
    public private(set) var sampleRate: Double = 16000

    public init() {}

    // MARK: - Device selection

    private func builtInInputDeviceID() -> AudioDeviceID? {
        var propAddr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject),
                                            &propAddr, 0, nil, &size) == noErr else { return nil }
        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        var ids = [AudioDeviceID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                        &propAddr, 0, nil, &size, &ids) == noErr else { return nil }
        for id in ids {
            var inAddr = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyStreams,
                mScope: kAudioDevicePropertyScopeInput,
                mElement: kAudioObjectPropertyElementMain)
            var inSize: UInt32 = 0
            guard AudioObjectGetPropertyDataSize(id, &inAddr, 0, nil, &inSize) == noErr,
                  inSize > 0 else { continue }
            var nameAddr = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyDeviceNameCFString,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain)
            var nameRef: Unmanaged<CFString>? = nil
            var nameSize = UInt32(MemoryLayout<CFString?>.size)
            guard AudioObjectGetPropertyData(id, &nameAddr, 0, nil, &nameSize, &nameRef) == noErr,
                  let name = nameRef?.takeRetainedValue() as String?,
                  name.lowercased().contains("built-in") else { continue }
            return id
        }
        return nil
    }

    private var previousDefaultInput: AudioDeviceID = 0

    private func switchToBuiltInMic() {
        guard forceBuiltInMic, let deviceID = builtInInputDeviceID() else { return }
        var defAddr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var prev: AudioDeviceID = 0
        var prevSize = UInt32(MemoryLayout<AudioDeviceID>.size)
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                    &defAddr, 0, nil, &prevSize, &prev)
        previousDefaultInput = prev
        var dev = deviceID
        AudioObjectSetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                    &defAddr, 0, nil,
                                    UInt32(MemoryLayout<AudioDeviceID>.size), &dev)
    }

    private func restoreDefaultMic() {
        guard previousDefaultInput != 0 else { return }
        var defAddr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var dev = previousDefaultInput
        AudioObjectSetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                    &defAddr, 0, nil,
                                    UInt32(MemoryLayout<AudioDeviceID>.size), &dev)
        previousDefaultInput = 0
    }

    // MARK: - Capture

    /// Begin capturing. Throws if the engine could not start.
    public func start() throws {
        guard !isRecording else { return }

        engine = AVAudioEngine()
        switchToBuiltInMic()

        let input = engine.inputNode
        accumulator.removeAll(keepingCapacity: true)
        lastPartialSampleCount = 0

        // Voice processing (beamforming) is DISABLED because the ASR model was trained on
        // raw single-mic audio. Voice processing applies DSP that changes the frequency
        // response, which the model doesn't handle well. Fan noise is already in the
        // training data. VAD handles accidental triggers instead.
        try? input.setVoiceProcessingEnabled(false)

        let inputFormat = input.outputFormat(forBus: 0)
        sampleRate = 16000

        // Create the resampler from the hardware format to 16 kHz mono float32.
        converter = AVAudioConverter(from: inputFormat, to: outputFormat)

        // Request the smallest buffer size the hardware supports.
        // 64 frames at 48 kHz = 1.3 ms per callback.
        var bufferFrameSize = UInt32(64)
        var prop = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyBufferFrameSize,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        AudioObjectSetPropertyData(
            AudioObjectID(engine.outputNode.auAudioUnit.deviceID),
            &prop, 0, nil,
            UInt32(MemoryLayout<UInt32>.size), &bufferFrameSize)

        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 64, format: inputFormat) { [weak self] buffer, _ in
            guard let self else { return }
            self.queue.async { self.append(buffer: buffer) }
        }

        engine.prepare()
        do {
            try engine.start()
            isRecording = true
        } catch {
            input.removeTap(onBus: 0)
            throw error
        }
    }

    public func stop() {
        guard isRecording else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        isRecording = false
        converter = nil
        restoreDefaultMic()
        partialCallback = nil
    }

    /// Drain captured samples into a float32 mono `Data` buffer.
    /// Returns nil if no audio was captured.
    public func takeBuffer() -> Data? {
        var snapshot = Data()
        queue.sync {
            snapshot = accumulator
            accumulator.removeAll(keepingCapacity: true)
        lastPartialSampleCount = 0
        }
        guard !snapshot.isEmpty else { return nil }
        return snapshot
    }

    // MARK: - Private

    private func append(buffer: AVAudioPCMBuffer) {
        guard buffer.floatChannelData?[0] != nil else { return }
        let n = Int(buffer.frameLength)
        let channels = Int(buffer.format.channelCount)
        let nativeFormat = buffer.format

        // Mix multi-channel to mono at the native sample rate.
        let monoBuffer: AVAudioPCMBuffer
        if channels == 1 {
            monoBuffer = buffer
        } else {
            guard let mixBuf = AVAudioPCMBuffer(pcmFormat: nativeFormat, frameCapacity: buffer.frameCapacity) else {
                return
            }
            mixBuf.frameLength = buffer.frameLength
            let mixDest = mixBuf.floatChannelData![0]
            for i in 0..<n {
                var sum: Float = 0
                for c in 0..<channels {
                    sum += buffer.floatChannelData![c][i]
                }
                mixDest[i] = sum / Float(channels)
            }
            monoBuffer = mixBuf
        }

        // Convert from the native sample rate (e.g. 48 kHz) to 16 kHz.
        guard let converter = AVAudioConverter(from: nativeFormat, to: outputFormat) else {
            // Fallback: append raw mono data at native rate if conversion fails.
            accumulator.append(UnsafeBufferPointer(start: monoBuffer.floatChannelData![0],
                                                    count: Int(monoBuffer.frameLength)))
            return
        }

        let outputFrameCapacity = AVAudioFrameCount(
            Double(monoBuffer.frameLength) * outputFormat.sampleRate / nativeFormat.sampleRate) + 1
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrameCapacity) else {
            return
        }

        var error: NSError?
        let status = converter.convert(to: outBuf, error: &error) { inNumPackets, outStatus in
            outStatus.pointee = .haveData
            return monoBuffer
        }

        if status == .error {
            NSLog("[AudioCapture] conversion error: %@", error?.localizedDescription ?? "unknown")
            return
        }

        guard let outData = outBuf.floatChannelData?[0] else { return }
        let outLen = Int(outBuf.frameLength)
        accumulator.append(UnsafeBufferPointer(start: outData, count: outLen))
        partialAccumulator.append(UnsafeBufferPointer(start: outData, count: outLen))
        if let cb = partialCallback {
            let total = partialAccumulator.count / MemoryLayout<Float>.stride
            if total - lastPartialSampleCount >= partialIntervalSamples {
                lastPartialSampleCount = total
                cb(partialAccumulator)
            }
        }

    }
}

extension Data {
    /// Append raw float32 samples from an UnsafeBufferPointer.
    fileprivate mutating func append(_ ptr: UnsafeBufferPointer<Float>) {
        ptr.withMemoryRebound(to: UInt8.self) { self.append(contentsOf: $0) }
    }
}
