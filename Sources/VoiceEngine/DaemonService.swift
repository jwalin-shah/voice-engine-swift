import Foundation
import AppKit

// MARK: - DaemonService
// Persistent Python daemon process for LFM-based cleanup and autocomplete.
// Communicates via JSON-RPC over stdin/stdout (line-delimited JSON).

public actor DaemonService {
    public enum Status {
        case notLaunched
        case launching
        case ready
        case failed(Error)
    }

    public enum DaemonError: LocalizedError {
        case notRunning
        case timeout
        case rpcError(String)
        case processExited(Int32)
        case pythonNotFound
        public var errorDescription: String? {
            switch self {
            case .notRunning: return "Daemon is not running"
            case .timeout: return "Daemon request timed out"
            case .rpcError(let msg): return "Daemon RPC error: \(msg)"
            case .processExited(let code): return "Daemon exited with code \(code)"
            case .pythonNotFound: return "Python 3 not found"
            }
        }
    }

    public nonisolated(unsafe) var status: Status = .notLaunched
    public private(set) nonisolated(unsafe) var isDaemonAvailable = false
    public private(set) nonisolated(unsafe) var modelLoaded = false

    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var pendingRequests: [Int: CheckedContinuation<[String: Any], Error>] = [:]
    private var nextId = 1
    private var restartCount = 0
    private var readBuffer = ""
    private let maxRestarts = 3
    private let queue = DispatchQueue(label: "voice.daemon-read", qos: .utility)

    public init() {}

    public func launch(scriptPath: String? = nil) async throws {
        status = .launching

        // Find Python 3
        let pythonPath = findPython()
        guard !pythonPath.isEmpty else {
            let err = DaemonError.pythonNotFound
            status = .failed(err)
            throw err
        }

        // Find daemon script
        let scriptPath = scriptPath ?? findScript()
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            let err = DaemonError.rpcError("Script not found: \(scriptPath)")
            status = .failed(err)
            throw err
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pythonPath)
        proc.arguments = ["-u", scriptPath]

        let stdinP = Pipe()
        let stdoutP = Pipe()
        proc.standardInput = stdinP
        proc.standardOutput = stdoutP
        proc.standardError = FileHandle.standardError

        self.process = proc
        self.stdinPipe = stdinP
        self.stdoutPipe = stdoutP

        try proc.run()

        // Start reading stdout on background queue
        let handle = stdoutP.fileHandleForReading
        queue.async { [weak self] in
            Task { await self?.readLoop(handle: handle) }
        }

        // Wait for startup response (timeout: 15s)
        try await waitForReady(timeout: 15.0)
        restartCount = 0
        isDaemonAvailable = true
    }

    private func findPython() -> String {
        let candidates = [
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
            "/usr/bin/python3",
        ]
        for p in candidates {
            if FileManager.default.fileExists(atPath: p) {
                return p
            }
        }
        // Fallback: which python3 via shell (last resort)
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        task.arguments = ["python3"]
        let outPipe = Pipe()
        task.standardOutput = outPipe
        try? task.run()
        task.waitUntilExit()
        let data = outPipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    private func findScript() -> String {
        // Bundled path (for production)
        if let bundlePath = Bundle.main.path(forResource: "daemon", ofType: "py", inDirectory: "lfm_daemon") {
            return bundlePath
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        // Development paths (most likely first)
        let devPaths = [
            home.appendingPathComponent("projects/voice-engine-swift/lfm_daemon/daemon.py").path,
            home.appendingPathComponent("projects/machine-scratch/voice-engine-swift/lfm_daemon/daemon.py").path,
        ]
        for p in devPaths {
            if FileManager.default.fileExists(atPath: p) { return p }
        }
        let cwd = FileManager.default.currentDirectoryPath
        let cwdPath = (cwd as NSString).appendingPathComponent("lfm_daemon/daemon.py")
        if FileManager.default.fileExists(atPath: cwdPath) { return cwdPath }
        return ""
    }

    private func waitForReady(timeout: TimeInterval) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let resp = try? await sendRequest(method: "ping", params: [:], silently: true) {
                modelLoaded = (resp["model_loaded"] as? Bool) ?? false
                status = .ready
                return
            }
            try await Task.sleep(for: .milliseconds(200))
        }
        status = .failed(DaemonError.timeout)
        throw DaemonError.timeout
    }

    public func sendRequest(method: String, params: [String: Any] = [:], silently: Bool = false) async throws -> [String: Any] {
        guard let proc = process, proc.isRunning else {
            throw DaemonError.notRunning
        }
        let rid = nextId
        nextId += 1

        return try await withCheckedThrowingContinuation { continuation in
            pendingRequests[rid] = continuation
            var req: [String: Any] = ["id": rid, "method": method]
            if !params.isEmpty { req["params"] = params }
            guard let data = try? JSONSerialization.data(withJSONObject: req),
                  let jsonStr = String(data: data, encoding: .utf8) else {
                pendingRequests.removeValue(forKey: rid)
                continuation.resume(throwing: DaemonError.rpcError("JSON encode failed"))
                return
            }
            stdinPipe?.fileHandleForWriting.write((jsonStr + "\n").data(using: .utf8)!)

            // Timeout task
            if !silently {
                Task {
                    try await Task.sleep(for: .seconds(10))
                    if let cont = pendingRequests.removeValue(forKey: rid) {
                        cont.resume(throwing: DaemonError.timeout)
                    }
                }
            }
        }
    }

    private func readLoop(handle: FileHandle) {
        while true {
            let data = handle.availableData
            if data.isEmpty { break }  // EOF
            guard let str = String(data: data, encoding: .utf8) else { continue }
            readBuffer += str
            processLines()
        }
        // Process exited
        Task { [weak self] in
            await self?.handleProcessExit()
        }
    }

    private func processLines() {
        while let newlineIdx = readBuffer.firstIndex(of: "\n") {
            let line = String(readBuffer[..<newlineIdx])
            readBuffer = String(readBuffer[readBuffer.index(after: newlineIdx)...])
            guard !line.isEmpty else { continue }
            guard let data = line.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                continue
            }
            // Match by id
            if let rid = json["id"] as? Int, let continuation = pendingRequests.removeValue(forKey: rid) {
                if let error = json["error"] as? String {
                    continuation.resume(throwing: DaemonError.rpcError(error))
                } else if let result = json["result"] as? [String: Any] {
                    continuation.resume(returning: result)
                } else {
                    continuation.resume(returning: json)
                }
            } else if json["error"] != nil {
                // Unmatched error — log via stderr
                fputs("[DaemonService] Unmatched response: \(line)\n", stderr)
            }
        }
    }

    private func handleProcessExit() async {
        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        isDaemonAvailable = false

        // Fail all pending requests
        for (_, cont) in pendingRequests {
            cont.resume(throwing: DaemonError.notRunning)
        }
        pendingRequests.removeAll()

        // Auto-restart
        guard restartCount < maxRestarts else {
            status = .failed(DaemonError.rpcError("Exceeded max restarts"))
            return
        }
        restartCount += 1
        let backoff = UInt64(restartCount) * 2_000_000_000  // 2s, 4s, 6s
        try? await Task.sleep(nanoseconds: backoff)
        do {
            try await launch()
        } catch {
            NSLog("[DaemonService] Restart \(restartCount)/\(maxRestarts) failed: \(error)")
        }
    }

    public func shutdown() {
        process?.terminate()
        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        isDaemonAvailable = false
        for (_, cont) in pendingRequests {
            cont.resume(throwing: DaemonError.notRunning)
        }
        pendingRequests.removeAll()
    }
}
