// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "VoiceEngine",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "voice-engine", targets: ["voice"]),
        .executable(name: "voice-tests", targets: ["VoiceEngineTests"]),
    ],
    targets: [
        .target(
            name: "VoiceEngine",
            path: "Sources/VoiceEngine",
            cSettings: [.define("ACCELERATE_NEW_LAPACK"), .define("ACCELERATE_LAPACK_ILP64")],
            swiftSettings: [.swiftLanguageMode(.v5)],
            linkerSettings: [.linkedFramework("AppKit"), .linkedFramework("AVFoundation"), .linkedFramework("CoreML"), .linkedFramework("Accelerate"), .linkedFramework("IOKit"), .linkedFramework("CoreAudio")]
        ),
        .executableTarget(
            name: "voice",
            dependencies: ["VoiceEngine"],
            path: "Sources/voice"
        ),
        .executableTarget(
            name: "VoiceEngineTests",
            dependencies: ["VoiceEngine"],
            path: "Tests/Runner"
        ),
        .testTarget(
            name: "VoiceEngineSwiftPMTests",
            dependencies: ["VoiceEngine"],
            path: "Tests/SwiftPM"
        ),
    ]
)
