// conformer_encoder.cpp — FastConformer encoder implementation (map-based weights).
#include "conformer_encoder.h"
#include <cmath>
#include <iostream>

namespace parakeet {

ConformerEncoder::ConformerEncoder(const EncoderConfig& cfg) : cfg_(cfg), pos_emb_(mx::zeros({1, 1, 1}, mx::float32)) {
    std::cout << "[ConformerEncoder] " << cfg_.n_layers << " layers, "
              << cfg_.d_model << " dim, " << cfg_.n_heads << " heads, "
              << "subsampling=" << cfg_.subsampling_factor << std::endl;
}

void ConformerEncoder::load_weights(
    const std::unordered_map<std::string, mx::array>& weights) {
    w_ = weights;  // copy the map
    pos_emb_ = compute_pos_emb(cfg_.pos_emb_max_len);
    mx::eval(pos_emb_);
    std::cout << "[ConformerEncoder] Loaded " << cfg_.n_layers << " layers of weights" << std::endl;
}

// --- Activation helpers ---

mx::array ConformerEncoder::silu(const mx::array& x) const {
    return mx::multiply(x, mx::sigmoid(x));
}

mx::array ConformerEncoder::glu(const mx::array& x) const {
    int d = x.shape().back();
    int half = d / 2;
    int B = x.shape()[0];
    int T = x.shape()[1];
    auto a = mx::slice(x, {0, 0, 0}, {B, T, half});
    auto b = mx::slice(x, {0, 0, half}, {B, T, d});
    return mx::multiply(a, mx::sigmoid(b));
}

// --- Linear layer ---

mx::array ConformerEncoder::linear(
    const mx::array& x, const std::string& weight_key,
    const std::string& bias_key) const {
    auto weight = W(weight_key);  // (out_features, in_features)
    if (has(bias_key)) {
        auto bias = W(bias_key);
        return mx::add(mx::matmul(x, mx::transpose(weight)), bias);
    }
    return mx::matmul(x, mx::transpose(weight));
}

// --- Layer norm ---

mx::array ConformerEncoder::layer_norm(
    const mx::array& x, const std::string& prefix) const {
    auto weight = W(prefix + ".weight");
    auto bias   = W(prefix + ".bias");
    return mx::fast::layer_norm(x, weight, bias, 1e-5f);
}

// --- Batch norm ---

mx::array ConformerEncoder::batch_norm(
    const mx::array& x, const std::string& prefix) const {
    auto weight = W(prefix + ".weight");
    auto bias   = W(prefix + ".bias");
    auto mean   = W(prefix + ".running_mean");
    auto var    = W(prefix + ".running_var");
    float eps = 1e-5f;
    auto rmean = mx::expand_dims(mx::expand_dims(mean, 0), 0);
    auto rvar  = mx::expand_dims(mx::expand_dims(var, 0), 0);
    auto w     = mx::expand_dims(mx::expand_dims(weight, 0), 0);
    auto b     = mx::expand_dims(mx::expand_dims(bias, 0), 0);
    auto normalized = mx::divide(
        mx::subtract(x, rmean),
        mx::sqrt(mx::add(rvar, mx::array(eps))));
    return mx::add(mx::multiply(normalized, w), b);
}

// --- Relative positional shift ---

mx::array ConformerEncoder::rel_shift(const mx::array& x) const {
    auto padded = mx::pad(x, {{0,0},{0,0},{0,0},{1,0}});
    int B = x.shape()[0], H = x.shape()[1];
    int Tq = x.shape()[2], pos_len = x.shape()[3];
    auto reshaped = mx::reshape(padded, {B, H, pos_len + 1, Tq});
    auto sliced = mx::slice(reshaped, {0, 0, 1, 0}, {B, H, pos_len + 1, Tq});
    return mx::reshape(sliced, {B, H, Tq, pos_len});
}

// --- Positional encoding ---

mx::array ConformerEncoder::compute_pos_emb(int max_len) const {
    int d = cfg_.d_model;
    int half_d = d / 2;
    int total_len = 2 * max_len - 1;

    // Create position array manually to avoid arange off-by-one
    std::vector<float> pos_data(total_len);
    for (int i = 0; i < total_len; i++) {
        pos_data[i] = static_cast<float>(max_len - 1 - i);
    }
    auto pos = mx::array(pos_data.data(), {total_len, 1}, mx::float32);
    auto div = mx::arange(0, half_d, mx::float32);
    div = mx::multiply(div, mx::array(-std::log(10000.0) / half_d));
    div = mx::exp(div);
    auto sin_vals = mx::sin(mx::multiply(pos, div));
    auto cos_vals = mx::cos(mx::multiply(pos, div));
    std::cout << "  PE sin_vals shape: [" << sin_vals.shape()[0] << ", " << sin_vals.shape()[1] << "]" << std::endl;

    std::vector<mx::array> cols;
    int sv0 = sin_vals.shape()[0];
    for (int i = 0; i < half_d; i++) {
        auto s = mx::reshape(mx::slice(sin_vals, {0, i}, {sv0, i + 1}), {total_len});
        auto c = mx::reshape(mx::slice(cos_vals, {0, i}, {sv0, i + 1}), {total_len});
        cols.push_back(s);
        cols.push_back(c);
    }
    std::cout << "  PE cols: " << cols.size() << " arrays of size " << cols[0].shape()[0] << std::endl;

    auto pe = mx::stack(cols, 1);
    std::cout << "  PE stacked shape: [" << pe.shape()[0] << ", " << pe.shape()[1] << "]" << std::endl;
    pe = mx::expand_dims(pe, 0);
    std::cout << "  PE final shape: [" << pe.shape()[0] << ", " << pe.shape()[1] << ", " << pe.shape()[2] << "]" << std::endl;
    return pe;
}

// --- Feed-forward ---

mx::array ConformerEncoder::feed_forward(
    const mx::array& x, int layer_idx, int ff_idx) const {
    std::string base = "encoder.layers." + std::to_string(layer_idx) +
                       (ff_idx == 0 ? ".feed_forward1" : ".feed_forward2");
    auto h = linear(x, base + ".linear1.weight", base + ".linear1.bias");
    h = silu(h);
    h = linear(h, base + ".linear2.weight", base + ".linear2.bias");
    return h;
}

// --- Self-attention with relative position ---

mx::array ConformerEncoder::multi_head_self_attention(
    const mx::array& x, int layer_idx, const mx::array& pos_emb) const {

    std::string base = "encoder.layers." + std::to_string(layer_idx) + ".self_attn";
    int B = x.shape()[0], T = x.shape()[1];
    int D = cfg_.d_model, H = cfg_.n_heads, Dh = D / H;

    auto q = linear(x, base + ".linear_q.weight", base + ".linear_q.bias");
    auto k = linear(x, base + ".linear_k.weight", base + ".linear_k.bias");
    auto v = linear(x, base + ".linear_v.weight", base + ".linear_v.bias");

    // Position embedding projection
    auto pos_w = W(base + ".linear_pos.weight");
    auto p = mx::matmul(pos_emb, mx::transpose(pos_w));

    // Reshape to [B, H, T, Dh]
    q = mx::reshape(q, {B, T, H, Dh});
    k = mx::reshape(k, {B, T, H, Dh});
    v = mx::reshape(v, {B, T, H, Dh});

    // Position biases
    auto pos_bias_u = W(base + ".pos_bias_u");  // [H, Dh]
    auto pos_bias_v = W(base + ".pos_bias_v");
    auto pb_u = mx::reshape(pos_bias_u, {1, H, 1, Dh});
    auto pb_v = mx::reshape(pos_bias_v, {1, H, 1, Dh});

    auto q_bh = mx::transpose(q, {0, 2, 1, 3});
    auto k_bh = mx::transpose(k, {0, 2, 1, 3});
    auto v_bh = mx::transpose(v, {0, 2, 1, 3});
    auto q_u = mx::add(q_bh, pb_u);
    auto q_v = mx::add(q_bh, pb_v);

    int pos_len = p.shape()[1];
    auto p_bh = mx::reshape(p, {1, pos_len, H, Dh});
    p_bh = mx::transpose(p_bh, {0, 2, 1, 3});
    if (p_bh.shape()[0] == 1 && B > 1) {
        p_bh = mx::broadcast_to(p_bh, {B, p_bh.shape()[1], p_bh.shape()[2], p_bh.shape()[3]});
    }

    // Manual attention: AC + BD (C++ scaled_dot_product_attention doesn't support additive masks)
    float scale = 1.0f / std::sqrt(static_cast<float>(Dh));

    // AC = q_u @ k^T * scale
    auto k_t = mx::transpose(k_bh, {0, 1, 3, 2});  // [B, H, Dh, Tk]
    auto ac = mx::multiply(mx::matmul(q_u, k_t), mx::array(scale));

    // BD = q_v @ p^T with rel_shift
    auto p_t = mx::transpose(p_bh, {0, 1, 3, 2});  // [B, H, Dh, pos_len]
    auto bd = mx::matmul(q_v, p_t);                   // [B, H, Tq, pos_len]
    bd = rel_shift(bd);
    bd = mx::slice(bd, {0, 0, 0, 0}, {B, H, T, T});
    bd = mx::multiply(bd, mx::array(scale));

    // Scores = AC + BD, then softmax, then @ V
    auto scores = mx::add(ac, bd);

    // Softmax
    auto attn_weights = mx::softmax(scores, -1);

    // Attn @ V
    auto attn_out = mx::matmul(attn_weights, v_bh);  // [B, H, Tq, Dh]

    // Reshape back: [B, H, T, Dh] -> [B, T, D]
    attn_out = mx::transpose(attn_out, {0, 2, 1, 3});
    attn_out = mx::reshape(attn_out, {B, T, D});

    // Output projection
    attn_out = linear(attn_out, base + ".linear_out.weight", base + ".linear_out.bias");
    return attn_out;
}

// --- Convolution block ---

mx::array ConformerEncoder::convolution_block(
    const mx::array& x, int layer_idx) const {

    std::string base = "encoder.layers." + std::to_string(layer_idx) + ".conv";

    // Pointwise conv1: 1x1 conv, (out_c, 1, in_c) -> (2*D, 1, D)
    auto pw1_w = W(base + ".pointwise_conv1.weight");
    auto h = mx::conv1d(x, pw1_w, 1, 0);

    // GLU
    h = glu(h);

    // Pad for depthwise conv
    int ksize = cfg_.conv_kernel_size;
    int padding = (ksize - 1) / 2;
    h = mx::pad(h, {{0, 0}, {padding, padding}, {0, 0}});

    // Depthwise conv (groups = d_model)
    auto dw_w = W(base + ".depthwise_conv.weight");
    h = mx::conv1d(h, dw_w, 1, 0, 1, cfg_.d_model);

    // Batch norm
    h = batch_norm(h, base + ".batch_norm");

    // SiLU
    h = silu(h);

    // Pointwise conv2
    auto pw2_w = W(base + ".pointwise_conv2.weight");
    h = mx::conv1d(h, pw2_w, 1, 0);

    return h;
}

// --- Pre-encode (subsampling) ---

static std::pair<mx::array, mx::array> dw_striding_subsampling(
    const mx::array& x,
    const mx::array& lengths,
    const std::unordered_map<std::string, mx::array>& w,
    int sampling_stages) {

    auto h = mx::expand_dims(x, -1); // [B, T, n_mels, 1]
    const int stride2 = 2, pad1 = 1, stride1 = 1, pad0 = 0;
    const int ksize = 3;  // kernel_size for subsampling convs

    // Update lengths — Python: floor((lengths + 2*pad - ksize) / stride) + 1
    auto len = mx::astype(lengths, mx::float32);
    for (int s = 0; s < sampling_stages; s++) {
        auto f_len = mx::floor(
            mx::divide(
                mx::add(len, mx::array(2 * pad1 - ksize)),
                mx::array(stride2)));
        len = mx::add(f_len, mx::array(1.0f));
    }
    len = mx::astype(len, mx::int32);

    auto do_conv = [&](const std::string& key, int stride, int pad, int groups,
                        bool apply_relu) {
        auto cw = w.at(key + ".weight");
        auto cb = w.at(key + ".bias");
        h = mx::add(
            mx::conv2d(h, cw, {stride, stride}, {pad, pad}, {1, 1}, groups),
            mx::reshape(cb, {1, 1, 1, -1}));
        if (apply_relu) {
            h = mx::maximum(h, mx::array(0.0f)); // ReLU
        }
    };

    // [B, T', F', C] -> [B, T', C, F'] -> [B, T', C*F']

    // Python conv list: [Conv2d_0, ReLU, Conv2d_2(dw), Conv2d_3(pw), ReLU,
    //                     Conv2d_5(dw), Conv2d_6(pw), ReLU]
    // ReLU only after conv0, conv3, conv6 — NOT after depthwise convs
    do_conv("encoder.pre_encode.conv.0", stride2, pad1, 1, true);   // conv0 + ReLU
    int ch = h.shape()[3];
    do_conv("encoder.pre_encode.conv.2", stride2, pad1, ch, false); // dw conv, no ReLU
    do_conv("encoder.pre_encode.conv.3", stride1, pad0, 1, true);   // pw conv + ReLU
    do_conv("encoder.pre_encode.conv.5", stride2, pad1, ch, false); // dw conv, no ReLU
    do_conv("encoder.pre_encode.conv.6", stride1, pad0, 1, true);   // pw conv + ReLU

    // [B, T', F', C] -> [B, T', C, F'] -> [B, T', C*F']
    h = mx::transpose(h, {0, 1, 3, 2});
    h = mx::reshape(h, {h.shape()[0], h.shape()[1], -1});

    // Linear projection
    auto out_w = w.at("encoder.pre_encode.out.weight");
    auto out_b = w.at("encoder.pre_encode.out.bias");
    h = mx::add(mx::matmul(h, mx::transpose(out_w)), out_b);

    return {h, len};
}

// --- Main forward pass ---

std::pair<mx::array, mx::array> ConformerEncoder::forward(const mx::array& mel) {
    int B = mel.shape()[0], T_in = mel.shape()[1];
    auto lengths = mx::full({B}, T_in, mx::int32);

    // Pre-encode (subsampling)
    // subsampling_factor = 2^stages
    int f = cfg_.subsampling_factor;
    int stages = 0;
    while (f > 1) { f >>= 1; stages++; }
    auto [x, out_lengths] = dw_striding_subsampling(mel, lengths, w_, stages);

    // Scale input (xscaling=False in config, so no scaling needed)
    int T = x.shape()[1];

    int max_len = cfg_.pos_emb_max_len;
    int center = max_len - 1;
    int pos_start = std::max(0, center - (T - 1));
    int pos_end = std::min(2 * max_len - 1, center + (T - 1) + 1);
    auto pe = mx::slice(pos_emb_, {0, pos_start, 0}, {1, pos_end, cfg_.d_model});
    if (pe.shape()[0] == 1 && B > 1) {
        pe = mx::broadcast_to(pe, {B, pe.shape()[1], pe.shape()[2]});
    }

    // Conformer blocks
    for (int i = 0; i < cfg_.n_layers; i++) {
        std::string layer_base = "encoder.layers." + std::to_string(i);

        // FF1 (0.5x residual)
        auto ff1_out = feed_forward(
            layer_norm(x, layer_base + ".norm_feed_forward1"), i, 0);
        x = mx::add(x, mx::multiply(ff1_out, mx::array(0.5f)));

        // Self-attention
        auto x_norm = layer_norm(x, layer_base + ".norm_self_att");
        auto attn_out = multi_head_self_attention(x_norm, i, pe);
        x = mx::add(x, attn_out);

        // Convolution
        auto x_conv_norm = layer_norm(x, layer_base + ".norm_conv");
        auto conv_out = convolution_block(x_conv_norm, i);
        x = mx::add(x, conv_out);

        // FF2 (0.5x residual)
        auto ff2_out = feed_forward(
            layer_norm(x, layer_base + ".norm_feed_forward2"), i, 1);
        x = mx::add(x, mx::multiply(ff2_out, mx::array(0.5f)));

        // Output norm
        x = layer_norm(x, layer_base + ".norm_out");
    }

    mx::eval(x, out_lengths);
    return {x, out_lengths};
}

} // namespace parakeet
