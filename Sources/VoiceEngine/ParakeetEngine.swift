// ParakeetEngine.swift — MLX Parakeet v2 inference via persistent Python subprocess.
//
// Architecture:
//   Swift (AVFoundation)  -->  audio path via stdin JSON  -->  Python worker (MLX)
//                                                              |
//   text result  <--  stdout JSON  <-----------------------------+
//
// The worker stays loaded between calls, eliminating cold-start overhead.
// IPC overhead measured at ~0.1ms per call.

import Foundation

public enum ParakeetError: LocalizedError {
    case notRunning
    case workerNotFound(String)
    case workerFailed(String)
    case timeout(String)
    case emptyAudio

    public var errorDescription: String? {
        switch self {
        case .notRunning: return "Worker process is not running. Call start() first."
        case .workerNotFound(let m): return "Worker script not found: \(m)"
        case .workerFailed(let m): return "Worker failed: \(m)"
        case .timeout(let m): return "Timeout waiting for worker: \(m)"
        case .emptyAudio: return "Audio buffer is empty"
        }
    }
}

public final class ParakeetEngine: @unchecked Sendable {

    // MARK: - Configuration

    private let workerScript: URL
    private let pythonBin: URL

    // MARK: - State

    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var isRunning = false
    private let queue = DispatchQueue(label: "com.voice-engine.parakeet-worker")

    // MARK: - Init

    /// - Parameters:
    ///   - workerScript: Path to `parakeet_worker.py`.
    ///   - pythonBin: Path to the Python 3 binary in the venv containing parakeet-mlx.
    public init(
        workerScript: URL? = nil,
        pythonBin: URL? = nil
    ) {
        let base = URL(fileURLWithPath: "/Users/jwalinshah/projects/voice-engine-swift")

        self.workerScript = workerScript
            ?? base.appendingPathComponent("Scripts/parakeet_worker.py")

        self.pythonBin = pythonBin
            ?? base.appendingPathComponent(".venv/bin/python3")
    }

    // MARK: - Lifecycle

    /// Start the persistent Python worker. Blocks until "ready" signal or timeout.
    /// Call once at app startup.
    public func start(timeoutSec: Double = 60) throws {
        try queue.sync {
            guard !isRunning else { return }

            // Validate paths
            guard FileManager.default.fileExists(atPath: workerScript.path) else {
                throw ParakeetError.workerNotFound(workerScript.path)
            }
            guard FileManager.default.fileExists(atPath: pythonBin.path) else {
                throw ParakeetError.workerNotFound(pythonBin.path)
            }

            let proc = Process()
            proc.executableURL = pythonBin
            proc.arguments = ["-u", workerScript.path]  // -u = unbuffered

            let stdinP = Pipe()
            let stdoutP = Pipe()
            let stderrP = Pipe()
            proc.standardInput = stdinP
            proc.standardOutput = stdoutP
            proc.standardError = stderrP

            // Collect stderr for logging and "ready" detection
            var stderrAccum = ""
            let readyGroup = DispatchGroup()
            readyGroup.enter()

            stderrP.fileHandleForReading.readabilityHandler = { [weak self] handle in
                let data = handle.availableData
                guard !data.isEmpty, let str = String(data: data, encoding: .utf8) else { return }
                stderrAccum += str
                // Log each line
                for line in str.split(separator: "\n", omittingEmptySubsequences: true) {
                    fflush(stdout)
                    print("[parakeet-worker] \(line)")
                }
                if stderrAccum.contains("ready") {
                    stderrAccum = ""
                    readyGroup.leave()
                }
            }

            try proc.run()

            self.process = proc
            self.stdinPipe = stdinP
            self.stdoutPipe = stdoutP
            self.stderrPipe = stderrP
            self.isRunning = true

            // Wait for "ready" signal
            let result = readyGroup.wait(timeout: .now() + timeoutSec)
            if result == .timedOut {
                // Check if process already died
                if !proc.isRunning {
                    let stderr = stderrAccum
                    proc.terminate()
                    self.isRunning = false
                    throw ParakeetError.workerFailed("Worker exited before ready. Stderr: \(stderr)")
                }
                // If still running but no ready signal, it might be loading
                // For parakeet-mlx, loading takes ~600ms. Wait a bit more.
            }

            // Give it a moment for stderr flush
            Thread.sleep(forTimeInterval: 2.0)
        }
    }

    /// Stop the worker process.
    public func stop() {
        queue.sync {
            guard isRunning, let proc = process else { return }
            isRunning = false

            // Send exit command
            if let stdinHandle = stdinPipe?.fileHandleForWriting {
                try? stdinHandle.write(contentsOf: "EXIT\n".data(using: .utf8)!)
                try? stdinHandle.close()
            }

            // Wait briefly then force-kill
            DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
                if proc.isRunning {
                    proc.terminate()
                }
            }
        }
    }

    deinit {
        stop()
    }

    // MARK: - Transcription

    /// Transcribe a WAV file. Worker must be running (call `start()` first).
    /// - Parameter wavPath: Absolute path to a 16kHz mono WAV file.
    /// - Returns: Transcribed text.
    public func transcribe(wavPath: String, timeoutSec: Double = 30) throws -> String {
        let req = """
        {"path": "\(wavPath)"}

        """

        guard isRunning,
              let stdinHandle = stdinPipe?.fileHandleForWriting,
              let stdoutHandle = stdoutPipe?.fileHandleForReading else {
            throw ParakeetError.notRunning
        }

        // Write request
        guard let reqData = req.data(using: .utf8) else {
            throw ParakeetError.workerFailed("Failed to encode request JSON")
        }
        try stdinHandle.write(contentsOf: reqData)

        // Read response (single line)
        let responseData = try stdoutHandle.readline(timeout: timeoutSec)

        guard let responseStr = String(data: responseData, encoding: .utf8),
              let responseJson = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any],
              let text = responseJson["text"] as? String else {
            throw ParakeetError.workerFailed("Invalid response: \(String(data: responseData, encoding: .utf8) ?? "nil")")
        }

        if let error = responseJson["error"] as? String, !text.isEmpty == false {
            throw ParakeetError.workerFailed(error)
        }

        return text
    }

    // MARK: - Status

    public var running: Bool { queue.sync { isRunning } }
}

// MARK: - FileHandle helpers

extension FileHandle {
    /// Read one newline-delimited line with a timeout.
    func readline(timeout: Double) throws -> Data {
        let deadline = Date().addingTimeInterval(timeout)
        var data = Data()

        while Date() < deadline {
            let available = try self.read(upToCount: 4096) ?? Data()
            if available.isEmpty {
                // EOF
                break
            }
            data.append(available)
            if data.contains(0x0A) {  // \n
                break
            }
            Thread.sleep(forTimeInterval: 0.001)  // 1ms polling
        }

        return data
    }
}
