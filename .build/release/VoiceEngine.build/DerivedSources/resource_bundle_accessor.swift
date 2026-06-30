import Foundation

extension Foundation.Bundle {
    static nonisolated let module: Bundle = {
        let mainPath = Bundle.main.bundleURL.appendingPathComponent("VoiceEngine_VoiceEngine.bundle").path
        let buildPath = "/Users/jwalinshah/projects/machine-scratch/voice-engine-swift/.build/arm64-apple-macosx/release/VoiceEngine_VoiceEngine.bundle"

        let preferredBundle = Bundle(path: mainPath)

        guard let bundle = preferredBundle ?? Bundle(path: buildPath) else {
            // Users can write a function called fatalError themselves, we should be resilient against that.
            Swift.fatalError("could not load resource bundle: from \(mainPath) or \(buildPath)")
        }

        return bundle
    }()
}