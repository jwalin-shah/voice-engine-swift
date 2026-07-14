// parakeet_model.h — Configuration structures for Parakeet-TDT model.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace parakeet {

struct PreprocessorConfig {
    int sample_rate = 16000;
    int n_mels = 128;
    int n_fft = 512;
    int hop_length = 160;   // 10ms at 16kHz
    int win_length = 400;   // 25ms at 16kHz
    std::string window_fn = "hann";
    std::string normalize = "per_feature";
    float preemph = 0.97f;
    float dither = 1e-5f;
};

struct EncoderConfig {
    int n_layers = 24;
    int d_model = 1024;
    int n_heads = 8;
    int ff_expansion_factor = 4;
    int conv_kernel_size = 9;
    int subsampling_factor = 8;
    int subsampling_conv_channels = 256;
    std::string self_attention_model = "rel_pos";
    bool use_bias = false;
    int pos_emb_max_len = 5000;
};

struct DecoderConfig {
    int pred_hidden = 640;
    int pred_rnn_layers = 2;
    int vocab_size = 1024;
    bool blank_as_pad = true;
};

struct JointConfig {
    int joint_hidden = 640;
    std::string activation = "relu";
    int num_extra_outputs = 5;  // TDT duration outputs
    int num_classes = 1024;     // vocabulary size
};

struct DecodingConfig {
    std::vector<int> durations = {0, 1, 2, 3, 4};
    int max_symbols = 10;
};

struct ParakeetConfig {
    PreprocessorConfig preprocessor;
    EncoderConfig encoder;
    DecoderConfig decoder;
    JointConfig joint;
    DecodingConfig decoding;

    // Derived values
    int blank_id() const { return joint.num_classes; }  // blank = vocab_size
    int num_outputs() const { return joint.num_classes + 1 + joint.num_extra_outputs; }
    int head_dim() const { return encoder.d_model / encoder.n_heads; }
    int ff_hidden_dim() const { return encoder.d_model * encoder.ff_expansion_factor; }
    int subsampling_stages() const {
        // subsampling_factor = 2^stages
        int stages = 0;
        int f = encoder.subsampling_factor;
        while (f > 1) { f >>= 1; stages++; }
        return stages;
    }

    // Load from model_config.json
    bool load(const std::string& path);
};

} // namespace parakeet
