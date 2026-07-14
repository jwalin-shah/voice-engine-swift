// parakeet_engine.h — Main Parakeet model class.
#pragma once

#include "mlx/mlx.h"
#include "parakeet_model.h"
#include "conformer_encoder.h"
#include "tdt_decoder.h"
#include "tokenizer.h"
#include "weight_loader.h"
#include <string>
#include <memory>

namespace parakeet {

namespace mx = mlx::core;

class ParakeetEngine {
public:
    struct Result {
        std::string text;
        double encoder_ms = 0;
        double decoder_ms = 0;
        double total_ms = 0;
    };

    // Load model from directory containing weights/ and config files
    bool load(const std::string& model_dir);

    // Transcribe audio file
    Result transcribe_file(const std::string& wav_path);
    Result transcribe_file_beam(const std::string& wav_path, int beam_width = 5);

    // Transcribe from raw float audio samples
    Result transcribe(const mx::array& audio);
    Result transcribe_beam(const mx::array& audio, int beam_width = 5);

    bool is_loaded() const { return loaded_; }

private:
    ParakeetConfig cfg_;
    std::unique_ptr<ConformerEncoder> encoder_;
    std::unique_ptr<TDTDecoder> decoder_;
    Tokenizer tokenizer_;
    std::string model_dir_;
    bool loaded_ = false;
};

} // namespace parakeet
