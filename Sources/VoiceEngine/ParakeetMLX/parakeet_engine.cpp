// parakeet_engine.cpp — Main Parakeet model implementation.
#include "parakeet_engine.h"
#include "audio_preprocess.h"

#include <chrono>
#include <iostream>
#include <fstream>

namespace parakeet {

static ParakeetConfig load_config(const std::string& config_path) {
    ParakeetConfig cfg;

    // Simple config parser — just reads the model_config.json
    // No JSON library needed for this simple flat format:
    // Use a minimal approach: read key=value pairs or use nlohmann
    // ponytail: for now, hardcode the known config from model_config.json
    // This avoids needing a JSON parser in the engine itself.

    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config: " + config_path);
    }

    // Read the whole file
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    // Simple string-based JSON parsing for flat keys
    auto extract_int = [&](const std::string& key, int default_val) -> int {
        std::string search = "\"" + key + "\": ";
        auto pos = content.find(search);
        if (pos == std::string::npos) return default_val;
        pos += search.length();
        return std::stoi(content.substr(pos));
    };

    auto extract_str = [&](const std::string& key, const std::string& default_val) -> std::string {
        std::string search = "\"" + key + "\": \"";
        auto pos = content.find(search);
        if (pos == std::string::npos) return default_val;
        pos += search.length();
        auto end = content.find("\"", pos);
        return content.substr(pos, end - pos);
    };

    auto extract_float = [&](const std::string& key, float default_val) -> float {
        std::string search = "\"" + key + "\": ";
        auto pos = content.find(search);
        if (pos == std::string::npos) return default_val;
        pos += search.length();
        return std::stof(content.substr(pos));
    };

    // Preprocessor
    cfg.preprocessor.sample_rate = extract_int("sample_rate", 16000);
    cfg.preprocessor.n_mels      = extract_int("n_mels", 128);
    cfg.preprocessor.n_fft       = extract_int("n_fft", 512);
    cfg.preprocessor.hop_length  = extract_int("hop_length", 160);
    cfg.preprocessor.win_length  = extract_int("win_length", 400);
    cfg.preprocessor.window_fn   = extract_str("window_fn", "hann");
    cfg.preprocessor.normalize   = extract_str("normalize", "per_feature");
    cfg.preprocessor.preemph     = extract_float("preemph", 0.97f);

    // Encoder
    cfg.encoder.n_layers            = extract_int("enc_n_layers", 24);
    cfg.encoder.d_model             = extract_int("enc_d_model", 1024);
    cfg.encoder.n_heads             = extract_int("enc_n_heads", 8);
    cfg.encoder.ff_expansion_factor = extract_int("enc_ff_expansion_factor", 4);
    cfg.encoder.conv_kernel_size    = extract_int("enc_conv_kernel_size", 9);
    cfg.encoder.subsampling_factor  = extract_int("enc_subsampling_factor", 8);
    cfg.encoder.subsampling_conv_channels = extract_int("enc_subsampling_conv_channels", 256);
    cfg.encoder.self_attention_model = extract_str("enc_self_attention_model", "rel_pos");
    cfg.encoder.pos_emb_max_len     = extract_int("enc_pos_emb_max_len", 5000);

    // Decoder
    cfg.decoder.pred_hidden     = extract_int("pred_hidden", 640);
    cfg.decoder.pred_rnn_layers = extract_int("pred_rnn_layers", 2);
    cfg.decoder.vocab_size      = extract_int("vocab_size", 1024);

    // Joint
    cfg.joint.joint_hidden    = extract_int("joint_hidden", 640);
    cfg.joint.activation      = extract_str("joint_activation", "relu");
    cfg.joint.num_extra_outputs = extract_int("num_extra_outputs", 5);
    cfg.joint.num_classes     = extract_int("num_classes", 1024);

    // Decoding
    cfg.decoding.durations = {0, 1, 2, 3, 4}; // fixed
    cfg.decoding.max_symbols = extract_int("max_symbols", 10);

    return cfg;
}

bool ParakeetEngine::load(const std::string& model_dir) {
    model_dir_ = model_dir;

    try {
        // Config files are in model_dir (alongside weights)
        std::string config_path = model_dir + "/model_config.json";
        cfg_ = load_config(config_path);
        std::cout << "[ParakeetEngine] Loaded config: "
                  << cfg_.encoder.n_layers << " layers, "
                  << cfg_.encoder.d_model << " dim, "
                  << cfg_.decoder.vocab_size << " vocab" << std::endl;

        // Load weights from model_dir (contains .f32 files + weight_entries.h)
        auto weights = WeightLoader::load(model_dir);

        // Initialize encoder
        encoder_ = std::make_unique<ConformerEncoder>(cfg_.encoder);
        encoder_->load_weights(weights);

        // Initialize decoder
        decoder_ = std::make_unique<TDTDecoder>(cfg_.decoder, cfg_.joint, cfg_.decoding);
        decoder_->load_weights(weights);

        // Initialize tokenizer
        std::string vocab_path = model_dir + "/vocab.txt";
        tokenizer_.load(vocab_path);

        loaded_ = true;
        std::cout << "[ParakeetEngine] Model loaded successfully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[ParakeetEngine] Failed to load: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

ParakeetEngine::Result ParakeetEngine::transcribe(const mx::array& audio) {
    Result result;
    if (!loaded_) {
        result.text = "[ERROR: model not loaded]";
        return result;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Audio preprocessing -> mel spectrogram
    auto mel = compute_log_mel(audio, cfg_.preprocessor, model_dir_);
    mx::eval(mel);
    auto t_mel = std::chrono::high_resolution_clock::now();

    // 2. Encoder forward
    auto [features, lengths] = encoder_->forward(mel);

    // bfloat16 roundtrip — matches Python transcribe() internal path
    features = mx::astype(features, mx::bfloat16);
    features = mx::astype(features, mx::float32);
    mx::eval(features, lengths);
    auto t_enc = std::chrono::high_resolution_clock::now();

    // 3. Decoder (beam=5)
    auto tokens = decoder_->decode_beam(features, lengths, 5);
    auto t_dec = std::chrono::high_resolution_clock::now();

    // 4. Tokenize -> text
    if (!tokens.empty() && !tokens[0].empty()) {
        result.text = tokenizer_.decode(tokens[0]);
    } else {
        result.text = "";
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    // Compute timing in ms
    using namespace std::chrono;
    result.encoder_ms = duration<double, std::milli>(t_enc - t_mel).count();
    result.decoder_ms = duration<double, std::milli>(t_dec - t_enc).count();
    result.total_ms   = duration<double, std::milli>(t_end - t_start).count();

    return result;
}

ParakeetEngine::Result ParakeetEngine::transcribe_file(const std::string& wav_path) {
    if (!loaded_) {
        Result r;
        r.text = "[ERROR: model not loaded]";
        return r;
    }

    auto audio = load_wav(wav_path, cfg_.preprocessor.sample_rate);
    return transcribe(audio);
}

ParakeetEngine::Result ParakeetEngine::transcribe_beam(const mx::array& audio, int beam_width) {
    Result result;
    if (!loaded_) {
        result.text = "[ERROR: model not loaded]";
        return result;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Audio preprocessing -> mel spectrogram
    auto mel = compute_log_mel(audio, cfg_.preprocessor, model_dir_);
    mx::eval(mel);

    auto t_mel = std::chrono::high_resolution_clock::now();

    // 2. Encoder forward
    auto [features, lengths] = encoder_->forward(mel);
    mx::eval(features, lengths);

    auto t_enc = std::chrono::high_resolution_clock::now();

    // 3. Decoder (beam search)
    auto tokens = decoder_->decode_beam(features, lengths, beam_width);
    auto t_dec = std::chrono::high_resolution_clock::now();

    // 4. Tokenize -> text
    if (!tokens.empty() && !tokens[0].empty()) {
        result.text = tokenizer_.decode(tokens[0]);
    } else {
        result.text = "";
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    using namespace std::chrono;
    result.encoder_ms = duration<double, std::milli>(t_enc - t_mel).count();
    result.decoder_ms = duration<double, std::milli>(t_dec - t_enc).count();
    result.total_ms   = duration<double, std::milli>(t_end - t_start).count();

    return result;
}

ParakeetEngine::Result ParakeetEngine::transcribe_file_beam(const std::string& wav_path, int beam_width) {
    if (!loaded_) {
        Result r;
        r.text = "[ERROR: model not loaded]";
        return r;
    }

    auto audio = load_wav(wav_path, cfg_.preprocessor.sample_rate);
    return transcribe_beam(audio, beam_width);
}

} // namespace parakeet
