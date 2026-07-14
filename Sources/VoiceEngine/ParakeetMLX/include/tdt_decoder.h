// tdt_decoder.h — TDT decoder (map-based weight storage).
#pragma once

#include "mlx/mlx.h"
#include "parakeet_model.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace parakeet {

namespace mx = mlx::core;

class TDTDecoder {
public:
    TDTDecoder(const DecoderConfig& dec_cfg, const JointConfig& joint_cfg,
               const DecodingConfig& decoding_cfg);

    void load_weights(const std::unordered_map<std::string, mx::array>& weights);
    std::vector<std::vector<int>> decode(
        const mx::array& encoder_features,
        const mx::array& encoder_lengths);

    std::vector<std::vector<int>> decode_beam(
        const mx::array& encoder_features,
        const mx::array& encoder_lengths,
        int beam_width = 5);

private:
    DecoderConfig dec_cfg_;
    JointConfig joint_cfg_;
    DecodingConfig decoding_cfg_;
    std::unordered_map<std::string, mx::array> w_;

    mx::array W(const std::string& key) const {
        auto it = w_.find(key);
        if (it == w_.end()) throw std::runtime_error("Missing weight: " + key);
        return it->second;
    }

    // LSTM forward step
    std::pair<mx::array, std::pair<mx::array, mx::array>> lstm_step(
        const mx::array& x,
        const mx::array& Wh, const mx::array& Wx, const mx::array& bias,
        const mx::array& h_prev, const mx::array& c_prev) const;

    // Predict network forward (single beam)
    struct PredictResult {
        mx::array output;
        mx::array h0, c0, h1, c1;
    };
    PredictResult predict(
        int token,
        const mx::array& h0, const mx::array& c0,
        const mx::array& h1, const mx::array& c1) const;

    // Predict network forward (zero-input, for initial beam)
    PredictResult predict_zero(
        const mx::array& h0, const mx::array& c0,
        const mx::array& h1, const mx::array& c1) const;

    // Predict network forward (batched beams, all tokens >= 0)
    struct BatchedPredictResult {
        mx::array output;  // [B, 1, pred_hidden]
        mx::array h0, c0, h1, c1;  // [B, pred_hidden]
    };
    BatchedPredictResult predict_batched(
        const std::vector<int>& tokens,
        const mx::array& h0, const mx::array& c0,
        const mx::array& h1, const mx::array& c1) const;

    // Joint network
    mx::array joint(
        const mx::array& enc_step,
        const mx::array& pred_out) const;
};

} // namespace parakeet
