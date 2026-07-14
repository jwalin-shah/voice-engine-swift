// conformer_encoder.h — FastConformer encoder for parakeet.
// Stores weights as a map (key -> array) since mx::array has no default ctor.
#pragma once

#include "mlx/mlx.h"
#include "parakeet_model.h"
#include <string>
#include <unordered_map>

namespace parakeet {

namespace mx = mlx::core;

class ConformerEncoder {
public:
    ConformerEncoder(const EncoderConfig& cfg);

    // Load weights from the weight map (key -> array)
    void load_weights(const std::unordered_map<std::string, mx::array>& weights);

    // Encode: mel spectrogram [B, T, n_mels] -> features [B, T', d_model], lengths [B]
    std::pair<mx::array, mx::array> forward(const mx::array& mel);

    // Helpers (public for bisector)
    mx::array feed_forward(const mx::array& x, int layer_idx, int ff_idx) const;
    mx::array multi_head_self_attention(
        const mx::array& x, int layer_idx, const mx::array& pos_emb) const;
    mx::array convolution_block(const mx::array& x, int layer_idx) const;
    mx::array layer_norm(const mx::array& x, const std::string& prefix) const;
    mx::array batch_norm(const mx::array& x, const std::string& prefix) const;
    mx::array linear(const mx::array& x, const std::string& weight_key,
                     const std::string& bias_key) const;
    mx::array rel_shift(const mx::array& x) const;
    mx::array compute_pos_emb(int max_len) const;
    mx::array silu(const mx::array& x) const;
    mx::array glu(const mx::array& x) const;

    // Weight access helper
    mx::array W(const std::string& key) const {
        auto it = w_.find(key);
        if (it == w_.end()) throw std::runtime_error("Missing weight: " + key);
        return it->second;
    }
    bool has(const std::string& key) const { return w_.find(key) != w_.end(); }

private:
    EncoderConfig cfg_;
    std::unordered_map<std::string, mx::array> w_;  // weight lookup

    // Precompute positional encoding (rel_pos)
    mx::array pos_emb_;
};

} // namespace parakeet
