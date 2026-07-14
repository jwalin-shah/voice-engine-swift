//  ParakeetMLXBridge.h
//  MLX C++ bridge for Parakeet v2 inference.
//
//  This header defines the C++ API that Swift calls through an
//  Objective-C++ bridging layer (ParakeetBridge.mm).
//
//  Architecture:
//    Swift (AVFoundation) -> ObjC++ -> MLX C++ -> libmlx.dylib -> Apple Silicon GPU
//
//  Build requirements:
//    - Link against libmlx.dylib (from mlx Python package)
//    - Include MLX C++ headers
//    - Framework: Metal, Foundation, QuartzCore
//    - C++17 or later
//
//  Status: SKELETON — the subprocess worker (Scripts/parakeet_worker.py) is the
//  working path. This bridge is the stretch goal for eliminating the remaining
//  Python graph-construction overhead (~20-25ms per transcribe call).

#pragma once

#include "mlx/mlx.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace parakeet {

// --- Configuration ---

struct ParakeetConfig {
    int sample_rate = 16000;
    int n_mels = 80;
    int n_fft = 512;
    int hop_length = 160;
    int win_length = 512;

    // Encoder
    int enc_hidden_dim = 512;
    int enc_num_blocks = 18;
    int enc_num_heads = 8;
    int enc_ff_dim = 2048;
    int enc_conv_kernel = 9;

    // Decoder
    int dec_hidden_dim = 512;
    int dec_vocab_size = 1024;
    int dec_num_layers = 4;

    // TDT-specific
    int max_duration = 4;
    int blank_id = 1024;
    int bos_id = 0;
    int eos_id = 2;

    // dtype
    mlx::core::Dtype dtype = mlx::core::float16;
};

// --- FastConformer Encoder ---

class FastConformerEncoder {
public:
    FastConformerEncoder(const ParakeetConfig& config);

    // Load weights from a safetensors file.
    void loadWeights(const std::string& safetensorsPath);

    // Preprocess: raw audio -> mel spectrogram.
    mlx::core::array computeMel(const mlx::core::array& audio);

    // Encode: mel spectrogram -> hidden states.
    // Returns (features, lengths).
    std::pair<mlx::core::array, mlx::core::array>
    encode(const mlx::core::array& mel);

private:
    ParakeetConfig cfg_;
    std::unordered_map<std::string, mlx::core::array> weights_;

    // Subsamping: depthwise strided convs
    mlx::core::array preEncode(const mlx::core::array& x, const mlx::core::array& lengths);

    // Single conformer block
    mlx::core::array conformerBlock(const mlx::core::array& x, int layerIdx);

    // Sub-components
    mlx::core::array feedForward(const mlx::core::array& x, int layerIdx, int ffIdx);
    mlx::core::array multiHeadAttention(const mlx::core::array& x, int layerIdx);
    mlx::core::array convolution(const mlx::core::array& x, int layerIdx);
};

// --- TDT Decoder ---

class TDTDecoder {
public:
    TDTDecoder(const ParakeetConfig& config);

    // Load weights from a safetensors file.
    void loadWeights(const std::string& safetensorsPath);

    // Autoregressive decode.
    // Returns token IDs (int array).
    mlx::core::array decode(
        const mlx::core::array& encoderFeatures,
        const mlx::core::array& encoderLengths,
        int maxSteps = 128
    );

private:
    ParakeetConfig cfg_;
    std::unordered_map<std::string, mlx::core::array> weights_;

    // Prediction network (embedding + LSTM/RNN)
    mlx::core::array predict(const mlx::core::array& tokens);

    // Joint network (combines encoder + predictor outputs)
    mlx::core::array joint(
        const mlx::core::array& encOut,
        const mlx::core::array& predOut
    );
};

// --- Top-level Parakeet Model ---

class ParakeetModel {
public:
    ParakeetModel(const std::string& modelDir);

    // Full transcribe: audio path -> text.
    // Returns transcribed text and elapsed ms.
    struct Result {
        std::string text;
        double totalMs;
    };

    Result transcribe(const std::string& wavPath);

private:
    ParakeetConfig cfg_;
    FastConformerEncoder encoder_;
    TDTDecoder decoder_;
    // tokenizer: SentencePiece (loaded from modelDir/tokenizer.model)

    // Load model config from config.json
    void loadConfig(const std::string& modelDir);

    // Load all weights from model.safetensors
    void loadWeights(const std::string& modelDir);
};

} // namespace parakeet
