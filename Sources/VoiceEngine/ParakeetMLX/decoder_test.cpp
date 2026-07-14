// decoder_test.cpp — Test decoder with Python-generated features.
#include "parakeet_model.h"
#include "tdt_decoder.h"
#include "tokenizer.h"
#include "weight_loader.h"
#include <fstream>
#include <iostream>

namespace mx = mlx::core;

int main() {
    std::string model_dir = "/Users/jwalinshah/projects/voice-engine-swift/Sources/VoiceEngine/ParakeetMLX/weights";

    // Load weights
    auto weights = parakeet::WeightLoader::load(model_dir);

    // Load tokenizer
    parakeet::Tokenizer tokenizer;
    tokenizer.load(model_dir + "/vocab.txt");

    // Create decoder
    parakeet::DecoderConfig dec_cfg;
    dec_cfg.vocab_size = 1024;
    dec_cfg.pred_hidden = 640;
    dec_cfg.pred_rnn_layers = 2;

    parakeet::JointConfig joint_cfg;
    joint_cfg.joint_hidden = 640;
    joint_cfg.activation = "relu";
    joint_cfg.num_extra_outputs = 5;
    joint_cfg.num_classes = 1024;

    parakeet::DecodingConfig decoding_cfg;
    decoding_cfg.durations = {0, 1, 2, 3, 4};
    decoding_cfg.max_symbols = 10;

    parakeet::TDTDecoder decoder(dec_cfg, joint_cfg, decoding_cfg);
    decoder.load_weights(weights);

    // Load Python-generated features
    std::ifstream feat_file("/tmp/parakeet_features.f32", std::ios::binary);
    feat_file.seekg(0, std::ios::end);
    size_t feat_size = feat_file.tellg() / sizeof(float);
    feat_file.seekg(0, std::ios::beg);
    std::vector<float> feat_data(feat_size);
    feat_file.read(reinterpret_cast<char*>(feat_data.data()), feat_size * sizeof(float));
    auto features_py = mx::array(feat_data.data(), {1, 64, 1024}, mx::float32);

    // Load C++ encoder features
    std::ifstream cpp_file("/tmp/parakeet_features_cpp.f32", std::ios::binary);
    cpp_file.seekg(0, std::ios::end);
    size_t cpp_feat_size = cpp_file.tellg() / sizeof(float);
    cpp_file.seekg(0, std::ios::beg);
    std::vector<float> cpp_feat_data(cpp_feat_size);
    cpp_file.read(reinterpret_cast<char*>(cpp_feat_data.data()), cpp_feat_size * sizeof(float));
    auto features_cpp = mx::array(cpp_feat_data.data(), {1, 64, 1024}, mx::float32);

    auto lengths = mx::full({1}, 64, mx::int32);

    // Test with Python features
    std::cout << "\n=== Python features ===" << std::endl;
    auto fmin_py = mx::min(features_py);
    auto fmax_py = mx::max(features_py);
    mx::eval(fmin_py, fmax_py);
    std::cout << "  min=" << fmin_py.item<float>() << " max=" << fmax_py.item<float>() << std::endl;

    auto tokens_py = decoder.decode(features_py, lengths);
    std::cout << "  Tokens: " << tokens_py[0].size() << std::endl;
    if (!tokens_py[0].empty()) {
        std::cout << "  Text: \"" << tokenizer.decode(tokens_py[0]) << "\"" << std::endl;
    }

    // Test with C++ features
    std::cout << "\n=== C++ features ===" << std::endl;
    auto fmin_cpp = mx::min(features_cpp);
    auto fmax_cpp = mx::max(features_cpp);
    mx::eval(fmin_cpp, fmax_cpp);
    std::cout << "  min=" << fmin_cpp.item<float>() << " max=" << fmax_cpp.item<float>() << std::endl;

    auto tokens_cpp = decoder.decode(features_cpp, lengths);
    std::cout << "  Tokens: " << tokens_cpp[0].size() << std::endl;
    if (!tokens_cpp[0].empty()) {
        std::cout << "  Text: \"" << tokenizer.decode(tokens_cpp[0]) << "\"" << std::endl;
    } else {
        std::cout << "  Empty output!" << std::endl;
    }

    // Check feature difference
    auto diff = mx::subtract(features_cpp, features_py);
    auto dmin = mx::min(diff);
    auto dmax = mx::max(diff);
    auto dmean = mx::mean(diff);
    auto dabs = mx::mean(mx::abs(diff));
    mx::eval(dmin, dmax, dmean, dabs);
    std::cout << "\n=== Feature difference (C++ - Python) ===" << std::endl;
    std::cout << "  min=" << dmin.item<float>() << " max=" << dmax.item<float>()
              << " mean=" << dmean.item<float>() << " mae=" << dabs.item<float>() << std::endl;

    return 0;
}
