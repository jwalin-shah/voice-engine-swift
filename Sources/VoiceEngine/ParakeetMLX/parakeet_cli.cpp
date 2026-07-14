// parakeet_cli.cpp — CLI test program for Parakeet C++ inference engine.
//
// Usage: parakeet_cli <model_dir> <wav_file> [--beam N]
//
// Build: cmake --build . --target parakeet_cli

#include "parakeet_engine.h"
#include <iostream>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    int beam_width = 0;  // 0 = greedy

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <wav_file> [--beam N]" << std::endl;
        std::cerr << "  model_dir: path to directory containing model_config.json, vocab.txt, and weight .f32 files" << std::endl;
        std::cerr << "  wav_file:  path to 16kHz mono WAV file" << std::endl;
        std::cerr << "  --beam N:  use beam search with width N (default: greedy)" << std::endl;
        return 1;
    }

    std::string model_dir = argv[1];
    std::string wav_path = argv[2];

    for (int i = 3; i < argc; i++) {
        if (std::strcmp(argv[i], "--beam") == 0 && i + 1 < argc) {
            beam_width = std::atoi(argv[++i]);
        }
    }

    std::cout << "=== Parakeet C++ Inference Engine ===" << std::endl;
    std::cout << "Model dir: " << model_dir << std::endl;
    std::cout << "WAV file:  " << wav_path << std::endl;
    std::cout << "Beam:      " << (beam_width > 0 ? std::to_string(beam_width) : "5 (default)") << std::endl;

    try {
        parakeet::ParakeetEngine engine;

        std::cout << "\nLoading model..." << std::endl;
        if (!engine.load(model_dir)) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }

        std::cout << "\nTranscribing..." << std::endl;
        auto result = beam_width > 0
            ? engine.transcribe_file_beam(wav_path, beam_width)
            : engine.transcribe_file(wav_path);

        std::cout << "\n=== Result ===" << std::endl;
        std::cout << "Text:       \"" << result.text << "\"" << std::endl;
        std::cout << "Encoder:    " << result.encoder_ms << " ms" << std::endl;
        std::cout << "Decoder:    " << result.decoder_ms << " ms" << std::endl;
        std::cout << "Total:      " << result.total_ms << " ms" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
