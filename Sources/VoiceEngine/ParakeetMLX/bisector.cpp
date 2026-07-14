// bisector.cpp — C++ encoder activation bisector.
// Loads the same mel from Python, runs encoder step-by-step,
// saves every intermediate tensor at named checkpoints matching Python's dump.
//
// Build: cmake --build . --target bisector
// Run:   ./bisector <model_dir> <fixture_dir>

#include "parakeet_model.h"
#include "conformer_encoder.h"
#include "weight_loader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

namespace mx = mlx::core;

// --- Save a float32 array to binary file ---
void save_f32(const std::string& path, const mx::array& arr) {
    auto a = mx::astype(arr, mx::float32);
    mx::eval(a);
    std::vector<float> data(a.size());
    // Copy to host
    a.eval();
    // Get raw pointer
    const float* ptr = a.data<float>();
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(ptr), a.size() * sizeof(float));
    out.close();
    std::cout << "  Saved: " << path << " [" << a.shape()[0];
    for (int i = 1; i < a.shape().size(); i++) std::cout << "," << a.shape()[i];
    std::cout << "]" << std::endl;
}

// --- Load float32 array from binary file ---
mx::array load_f32(const std::string& path, const std::vector<int>& shape) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open: " + path);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    auto arr = mx::array(data.data(), mx::Shape(shape.begin(), shape.end()), mx::float32);
    mx::eval(arr);
    return arr;
}

// --- Helper: print stats ---
void print_stats(const std::string& name, const mx::array& arr) {
    auto mn = mx::min(arr);
    auto mxv = mx::max(arr);
    mx::eval(mn, mxv);
    std::cout << "  " << name << " shape=[";
    for (int i = 0; i < arr.shape().size(); i++) {
        if (i > 0) std::cout << ",";
        std::cout << arr.shape()[i];
    }
    std::cout << "] min=" << mn.item<float>() << " max=" << mxv.item<float>() << std::endl;
}

int main(int argc, char** argv) {
    std::string model_dir = argc > 1 ? argv[1] : "Sources/VoiceEngine/ParakeetMLX/weights";
    std::string fixture_dir = argc > 2 ? argv[2] : "/tmp/activation_fixture";
    std::string out_dir = fixture_dir + "/cpp_dump";

    mkdir(out_dir.c_str(), 0755);

    std::cout << "=== C++ Encoder Activation Bisector ===" << std::endl;
    std::cout << "Model dir:  " << model_dir << std::endl;
    std::cout << "Fixture:    " << fixture_dir << std::endl;
    std::cout << "Output:     " << out_dir << std::endl;

    // 1. Load weights
    auto weights = parakeet::WeightLoader::load(model_dir);

    // 2. Create encoder
    parakeet::EncoderConfig enc_cfg;
    enc_cfg.n_layers = 24;
    enc_cfg.d_model = 1024;
    enc_cfg.n_heads = 8;
    enc_cfg.ff_expansion_factor = 4;
    enc_cfg.conv_kernel_size = 9;
    enc_cfg.subsampling_factor = 8;
    enc_cfg.subsampling_conv_channels = 256;
    enc_cfg.self_attention_model = "rel_pos";
    enc_cfg.use_bias = false;  // Note: Python parakeet uses bias=True for Conv2d!
    enc_cfg.pos_emb_max_len = 5000;

    parakeet::ConformerEncoder encoder(enc_cfg);
    encoder.load_weights(weights);

    // 3. Load mel from Python fixture (read shape dynamically)
    // Shape is stored as "B T F" in mel_shape.txt
    int B_mel, T_in, n_mels;
    {
        std::ifstream shape_file(fixture_dir + "/mel_shape.txt");
        if (!shape_file.is_open()) {
            throw std::runtime_error("Cannot open: " + fixture_dir + "/mel_shape.txt");
        }
        shape_file >> B_mel >> T_in >> n_mels;
        std::cout << "Mel shape from file: [" << B_mel << ", " << T_in << ", " << n_mels << "]" << std::endl;
    }
    auto mel = load_f32(fixture_dir + "/mel_input.f32", {B_mel, T_in, n_mels});
    print_stats("mel_input (from Python)", mel);

    int B = B_mel;
    auto lengths = mx::full({B}, T_in, mx::int32);

    save_f32(out_dir + "/encoder_mel_input.f32", mel);

    // 4. Replicate Python's pre-encode step by step
    // Python: expand_dims(x, axis=1) → [B, 1, T, n_mels]
    // Python conv_forward: transpose (0,2,3,1) → [B, T, n_mels, 1]
    // C++ equivalent: expand_dims(x, -1) → [B, T, n_mels, 1] (same shape, different order)
    // ACTUALLY: Python does expand_dims(axis=1) then transpose(0,2,3,1) to get [B,T,n_mels,1]
    // C++ just does expand_dims(-1) to get [B,T,n_mels,1]. Same result.

    auto x = mel;
    save_f32(out_dir + "/encoder_mel_input.f32", x);  // alias

    // Pre-encode expanded
    auto h = mx::expand_dims(x, -1);  // [B, T, n_mels, 1]
    save_f32(out_dir + "/pre_encode_expand_dims.f32", h);

    // conv_forward input: Python's [B, T, n_mels, 1] = C++'s [B, T, n_mels, 1]
    save_f32(out_dir + "/pre_encode_conv_forward_input.f32", h);

    // Python conv list: [Conv2d_0, ReLU, Conv2d_2(dw), Conv2d_3(pw), ReLU, Conv2d_5(dw), Conv2d_6(pw), ReLU]
    // Names: conv.0.Conv2d, conv.1.ReLU, conv.2.Conv2d, conv.3.Conv2d, conv.4.ReLU, ...

    const int stride2 = 2, pad1 = 1, stride1 = 1, pad0 = 0;
    const int ksize = 3;

    // ReLU layer indices in Python conv list:
    //   conv[0]=Conv2d, conv[1]=ReLU, conv[2]=Conv2d(dw), conv[3]=Conv2d(pw),
    //   conv[4]=ReLU, conv[5]=Conv2d(dw), conv[6]=Conv2d(pw), conv[7]=ReLU
    // We match Python's layer indices exactly for file comparison.

    // conv0: Conv2d(1→256, k=3, s=2, p=1) then ReLU at index 1
    {
        auto cw = weights.at("encoder.pre_encode.conv.0.weight");
        auto cb = weights.at("encoder.pre_encode.conv.0.bias");
        std::cout << "  conv.0 weight_shape=[";
        for (int i = 0; i < cw.shape().size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << cw.shape()[i];
        }
        std::cout << "] groups=1 stride=2" << std::endl;
        h = mx::add(
            mx::conv2d(h, cw, {stride2, stride2}, {pad1, pad1}, {1, 1}, 1),
            mx::reshape(cb, {1, 1, 1, -1}));
        mx::eval(h);
        save_f32(out_dir + "/pre_encode_conv_0_Conv2d.f32", h);
        print_stats("conv.0", h);
        h = mx::maximum(h, mx::array(0.0f));  // ReLU at index 1
        mx::eval(h);
        save_f32(out_dir + "/pre_encode_conv_1_ReLU.f32", h);
        print_stats("conv.1 (ReLU)", h);
    }

    // conv2: Conv2d(256→256, k=3, s=2, p=1, groups=256) — depthwise, no ReLU
    {
        int ch = h.shape()[3];
        auto cw = weights.at("encoder.pre_encode.conv.2.weight");
        auto cb = weights.at("encoder.pre_encode.conv.2.bias");
        h = mx::add(
            mx::conv2d(h, cw, {stride2, stride2}, {pad1, pad1}, {1, 1}, ch),
            mx::reshape(cb, {1, 1, 1, -1}));
        mx::eval(h);
        save_f32(out_dir + "/pre_encode_conv_2_Conv2d.f32", h);
        print_stats("conv.2 (dw)", h);
    }

    // conv3: Conv2d(256→256, k=1, s=1, p=0, groups=1) — pointwise + ReLU at index 4
    {
        auto cw = weights.at("encoder.pre_encode.conv.3.weight");
        auto cb = weights.at("encoder.pre_encode.conv.3.bias");
        h = mx::add(
            mx::conv2d(h, cw, {stride1, stride1}, {pad0, pad0}, {1, 1}, 1),
            mx::reshape(cb, {1, 1, 1, -1}));
        mx::eval(h);
        save_f32(out_dir + "/pre_encode_conv_3_Conv2d.f32", h);
        print_stats("conv.3 (pw)", h);
        h = mx::maximum(h, mx::array(0.0f));  // ReLU at index 4
        mx::eval(h);
        save_f32(out_dir + "/pre_encode_conv_4_ReLU.f32", h);
        print_stats("conv.4 (ReLU)", h);
    }

    // conv5: Conv2d(256→256, k=3, s=2, p=1, groups=256) — depthwise, no ReLU
    {
        int ch = h.shape()[3];
        auto cw = weights.at("encoder.pre_encode.conv.5.weight");
        auto cb = weights.at("encoder.pre_encode.conv.5.bias");
        h = mx::add(
            mx::conv2d(h, cw, {stride2, stride2}, {pad1, pad1}, {1, 1}, ch),
            mx::reshape(cb, {1, 1, 1, -1}));
        mx::eval(h);
        save_f32(out_dir + "/pre_encode_conv_5_Conv2d.f32", h);
        print_stats("conv.5 (dw)", h);
    }

    // conv6: Conv2d(256→256, k=1, s=1, p=0, groups=1) — pointwise + ReLU at index 7
    {
        auto cw = weights.at("encoder.pre_encode.conv.6.weight");
        auto cb = weights.at("encoder.pre_encode.conv.6.bias");
        h = mx::add(
            mx::conv2d(h, cw, {stride1, stride1}, {pad0, pad0}, {1, 1}, 1),
            mx::reshape(cb, {1, 1, 1, -1}));
        mx::eval(h);
        save_f32(out_dir + "/pre_encode_conv_6_Conv2d.f32", h);
        print_stats("conv.6 (pw)", h);
        h = mx::maximum(h, mx::array(0.0f));  // ReLU at index 7
        mx::eval(h);
        save_f32(out_dir + "/pre_encode_conv_7_ReLU.f32", h);
        print_stats("conv.7 (ReLU)", h);
    }

    // conv_forward output: transpose (0,3,1,2) — NHWC → NCHW
    // Python: x.transpose((0, 3, 1, 2)) → [B, C, T', F']
    auto h_nchw = mx::transpose(h, {0, 3, 1, 2});
    mx::eval(h_nchw);
    save_f32(out_dir + "/pre_encode_conv_forward_output.f32", h_nchw);
    print_stats("conv_forward.output (NCHW)", h_nchw);

    // Out lengths (same computation as Python)
    auto len = mx::astype(lengths, mx::float32);
    int sampling_stages = 3;  // subsampling_factor=8 = 2^3
    for (int s = 0; s < sampling_stages; s++) {
        auto f_len = mx::floor(
            mx::divide(
                mx::add(len, mx::array(2 * pad1 - ksize)),
                mx::array(stride2)));
        len = mx::add(f_len, mx::array(1.0f));
    }
    len = mx::astype(len, mx::int32);
    save_f32(out_dir + "/pre_encode_out_lengths.f32", len);
    print_stats("out_lengths", len);

    // Reshape: swapaxes(1,2) + reshape → [B, T', C*F']
    // Python: x.swapaxes(1, 2).reshape(x.shape[0], x.shape[2], -1)
    // swapaxes(1,2) of [B, C, T', F'] → [B, T', C, F']
    // reshape to [B, T', C*F']
    auto h_swapped = mx::transpose(h_nchw, {0, 2, 1, 3});  // swapaxes(1,2)
    int T_out = h_swapped.shape()[1];
    int C = h_swapped.shape()[2];
    int F = h_swapped.shape()[3];
    auto h_flat = mx::reshape(h_swapped, {h_swapped.shape()[0], T_out, C * F});
    mx::eval(h_flat);
    save_f32(out_dir + "/pre_encode_reshaped.f32", h_flat);
    print_stats("reshaped", h_flat);

    // Linear projection: self.out(x)
    auto out_w = weights.at("encoder.pre_encode.out.weight");
    auto out_bias = weights.at("encoder.pre_encode.out.bias");
    auto h_proj = mx::add(mx::matmul(h_flat, mx::transpose(out_w)), out_bias);
    mx::eval(h_proj);
    save_f32(out_dir + "/pre_encode_projection.f32", h_proj);
    print_stats("projection", h_proj);

    x = h_proj;

    // --- Positional encoding ---
    // xscaling=False → scale=1.0, no change
    save_f32(out_dir + "/pos_enc_scaled.f32", x);  // same as projection

    int T = x.shape()[1];
    int max_len = enc_cfg.pos_emb_max_len;
    int center = max_len - 1;
    int pos_start = std::max(0, center - (T - 1));
    int pos_end = std::min(2 * max_len - 1, center + (T - 1) + 1);

    // Access pos_emb_ via the precomputed member
    // We need a public interface — let's compute it directly
    auto pos_emb_full = encoder.compute_pos_emb(max_len);
    auto pe = mx::slice(pos_emb_full, {0, pos_start, 0}, {1, pos_end, enc_cfg.d_model});
    if (pe.shape()[0] == 1 && B > 1) {
        pe = mx::broadcast_to(pe, {B, pe.shape()[1], pe.shape()[2]});
    }
    mx::eval(pe);
    save_f32(out_dir + "/pos_enc_embedding.f32", pe);
    print_stats("pos_emb", pe);

    // --- Conformer blocks (first 2 with full detail) ---
    for (int i = 0; i < 2; i++) {
        std::string b = std::string("block") + (i < 10 ? "0" : "") + std::to_string(i);
        std::string layer_base = "encoder.layers." + std::to_string(i);

        save_f32(out_dir + "/" + b + "_input.f32", x);
        print_stats(b + ".input", x);

        // FF1: layer_norm → feed_forward → 0.5*residual
        auto norm_ff1 = encoder.layer_norm(x, layer_base + ".norm_feed_forward1");
        mx::eval(norm_ff1);
        save_f32(out_dir + "/" + b + "_ffn1_layer_norm_out.f32", norm_ff1);
        print_stats(b + ".ffn1.ln_out", norm_ff1);

        auto ff1_out = encoder.feed_forward(norm_ff1, i, 0);
        mx::eval(ff1_out);
        save_f32(out_dir + "/" + b + "_ffn1_linear_out.f32", ff1_out);
        print_stats(b + ".ffn1.out", ff1_out);

        auto ff1_scaled = mx::multiply(ff1_out, mx::array(0.5f));
        mx::eval(ff1_scaled);
        save_f32(out_dir + "/" + b + "_ffn1_scaled_0_5.f32", ff1_scaled);

        x = mx::add(x, ff1_scaled);
        mx::eval(x);
        save_f32(out_dir + "/" + b + "_ffn1_add.f32", x);
        print_stats(b + ".ffn1.add", x);

        // Self-attention
        auto norm_attn = encoder.layer_norm(x, layer_base + ".norm_self_att");
        mx::eval(norm_attn);
        save_f32(out_dir + "/" + b + "_attn_layer_norm_out.f32", norm_attn);
        print_stats(b + ".attn.ln_out", norm_attn);

        // Manual attention tracing
        std::string attn_base = layer_base + ".self_attn";
        int D = enc_cfg.d_model, H = enc_cfg.n_heads, Dh = D / H;

        auto q = encoder.linear(norm_attn, attn_base + ".linear_q.weight", attn_base + ".linear_q.bias");
        auto k = encoder.linear(norm_attn, attn_base + ".linear_k.weight", attn_base + ".linear_k.bias");
        auto v = encoder.linear(norm_attn, attn_base + ".linear_v.weight", attn_base + ".linear_v.bias");

        auto pos_w = encoder.W(attn_base + ".linear_pos.weight");
        auto p = mx::matmul(pe, mx::transpose(pos_w));
        mx::eval(q, k, v, p);
        save_f32(out_dir + "/" + b + "_attn_q.f32", q);
        save_f32(out_dir + "/" + b + "_attn_k.f32", k);
        save_f32(out_dir + "/" + b + "_attn_v.f32", v);
        save_f32(out_dir + "/" + b + "_attn_p.f32", p);

        // Reshape to heads
        auto q_heads = mx::reshape(q, {B, T, H, Dh});
        auto k_heads = mx::reshape(k, {B, T, H, Dh});
        auto v_heads = mx::reshape(v, {B, T, H, Dh});

        auto pos_bias_u = encoder.W(attn_base + ".pos_bias_u");
        auto pos_bias_v = encoder.W(attn_base + ".pos_bias_v");
        auto pb_u = mx::reshape(pos_bias_u, {1, H, 1, Dh});
        auto pb_v = mx::reshape(pos_bias_v, {1, H, 1, Dh});

        auto q_bh = mx::transpose(q_heads, {0, 2, 1, 3});
        auto k_bh = mx::transpose(k_heads, {0, 2, 1, 3});
        auto v_bh = mx::transpose(v_heads, {0, 2, 1, 3});
        auto q_u = mx::add(q_bh, pb_u);
        auto q_v = mx::add(q_bh, pb_v);
        mx::eval(q_u, q_v, k_bh, v_bh);
        save_f32(out_dir + "/" + b + "_attn_q_u.f32", q_u);
        save_f32(out_dir + "/" + b + "_attn_q_v.f32", q_v);
        save_f32(out_dir + "/" + b + "_attn_k_heads.f32", k_bh);
        save_f32(out_dir + "/" + b + "_attn_v_heads.f32", v_bh);

        int pos_len = p.shape()[1];
        auto p_heads = mx::reshape(p, {1, pos_len, H, Dh});
        p_heads = mx::transpose(p_heads, {0, 2, 1, 3});
        if (p_heads.shape()[0] == 1 && B > 1) {
            p_heads = mx::broadcast_to(p_heads, {B, p_heads.shape()[1], p_heads.shape()[2], p_heads.shape()[3]});
        }
        mx::eval(p_heads);
        save_f32(out_dir + "/" + b + "_attn_p_heads.f32", p_heads);

        // AC = q_u @ k^T * scale (AC only, no pos bias)
        float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
        auto k_t = mx::transpose(k_bh, {0, 1, 3, 2});
        auto ac = mx::multiply(mx::matmul(q_u, k_t), mx::array(scale));
        mx::eval(ac);
        save_f32(out_dir + "/" + b + "_attn_ac_only.f32", ac);

        // BD = q_v @ p^T with rel_shift
        auto p_t = mx::transpose(p_heads, {0, 1, 3, 2});
        auto bd = mx::matmul(q_v, p_t);
        bd = encoder.rel_shift(bd);
        bd = mx::slice(bd, {0, 0, 0, 0}, {B, H, T, T});
        bd = mx::multiply(bd, mx::array(scale));
        mx::eval(bd);
        save_f32(out_dir + "/" + b + "_attn_matrix_bd.f32", bd);

        // Full attention
        auto scores = mx::add(ac, bd);
        auto attn_weights = mx::softmax(scores, -1);
        auto attn_out = mx::matmul(attn_weights, v_bh);
        mx::eval(attn_out);
        save_f32(out_dir + "/" + b + "_attn_sdpa_out.f32", attn_out);

        // Reshape and output projection
        attn_out = mx::transpose(attn_out, {0, 2, 1, 3});
        attn_out = mx::reshape(attn_out, {B, T, D});
        attn_out = encoder.linear(attn_out, attn_base + ".linear_out.weight", attn_base + ".linear_out.bias");
        mx::eval(attn_out);
        save_f32(out_dir + "/" + b + "_attn_linear_out.f32", attn_out);

        x = mx::add(x, attn_out);
        mx::eval(x);
        save_f32(out_dir + "/" + b + "_attn_add.f32", x);
        print_stats(b + ".attn.add", x);

        // Convolution block
        auto conv_norm = encoder.layer_norm(x, layer_base + ".norm_conv");
        mx::eval(conv_norm);
        save_f32(out_dir + "/" + b + "_conv_layer_norm_out.f32", conv_norm);
        print_stats(b + ".conv.ln_out", conv_norm);

        std::string conv_base = layer_base + ".conv";

        // PW1
        auto pw1_w = encoder.W(conv_base + ".pointwise_conv1.weight");
        auto c = mx::conv1d(conv_norm, pw1_w, 1, 0);
        mx::eval(c);
        save_f32(out_dir + "/" + b + "_conv_pw1.f32", c);
        print_stats(b + ".conv.pw1", c);

        // GLU
        c = encoder.glu(c);
        mx::eval(c);
        save_f32(out_dir + "/" + b + "_conv_glu.f32", c);
        print_stats(b + ".conv.glu", c);

        // Pad + depthwise conv
        int ksize_conv = enc_cfg.conv_kernel_size;
        int padding = (ksize_conv - 1) / 2;
        c = mx::pad(c, {{0, 0}, {padding, padding}, {0, 0}});
        auto dw_w = encoder.W(conv_base + ".depthwise_conv.weight");
        c = mx::conv1d(c, dw_w, 1, 0, 1, enc_cfg.d_model);
        mx::eval(c);
        save_f32(out_dir + "/" + b + "_conv_dw.f32", c);
        print_stats(b + ".conv.dw", c);

        // Batch norm
        c = encoder.batch_norm(c, conv_base + ".batch_norm");
        mx::eval(c);
        save_f32(out_dir + "/" + b + "_conv_bn.f32", c);
        print_stats(b + ".conv.bn", c);

        // SiLU
        c = encoder.silu(c);
        mx::eval(c);
        save_f32(out_dir + "/" + b + "_conv_silu.f32", c);
        print_stats(b + ".conv.silu", c);

        // PW2
        auto pw2_w = encoder.W(conv_base + ".pointwise_conv2.weight");
        c = mx::conv1d(c, pw2_w, 1, 0);
        mx::eval(c);
        save_f32(out_dir + "/" + b + "_conv_pw2.f32", c);
        print_stats(b + ".conv.pw2", c);

        x = mx::add(x, c);
        mx::eval(x);
        save_f32(out_dir + "/" + b + "_conv_add.f32", x);
        print_stats(b + ".conv.add", x);

        // FF2
        auto norm_ff2 = encoder.layer_norm(x, layer_base + ".norm_feed_forward2");
        mx::eval(norm_ff2);
        save_f32(out_dir + "/" + b + "_ffn2_layer_norm_out.f32", norm_ff2);
        print_stats(b + ".ffn2.ln_out", norm_ff2);

        auto ff2_out = encoder.feed_forward(norm_ff2, i, 1);
        mx::eval(ff2_out);
        save_f32(out_dir + "/" + b + "_ffn2_linear_out.f32", ff2_out);
        print_stats(b + ".ffn2.out", ff2_out);

        x = mx::add(x, mx::multiply(ff2_out, mx::array(0.5f)));
        mx::eval(x);
        save_f32(out_dir + "/" + b + "_ffn2_add.f32", x);
        print_stats(b + ".ffn2.add", x);

        // Output norm
        x = encoder.layer_norm(x, layer_base + ".norm_out");
        mx::eval(x);
        save_f32(out_dir + "/" + b + "_out.f32", x);
        print_stats(b + ".out", x);
    }

    // Run remaining blocks without tracking
    for (int i = 2; i < enc_cfg.n_layers; i++) {
        std::string layer_base = "encoder.layers." + std::to_string(i);

        auto ff1_out = encoder.feed_forward(
            encoder.layer_norm(x, layer_base + ".norm_feed_forward1"), i, 0);
        x = mx::add(x, mx::multiply(ff1_out, mx::array(0.5f)));

        auto x_norm = encoder.layer_norm(x, layer_base + ".norm_self_att");
        auto attn_out = encoder.multi_head_self_attention(x_norm, i, pe);
        x = mx::add(x, attn_out);

        auto x_conv_norm = encoder.layer_norm(x, layer_base + ".norm_conv");
        auto conv_out = encoder.convolution_block(x_conv_norm, i);
        x = mx::add(x, conv_out);

        auto ff2_out = encoder.feed_forward(
            encoder.layer_norm(x, layer_base + ".norm_feed_forward2"), i, 1);
        x = mx::add(x, mx::multiply(ff2_out, mx::array(0.5f)));

        x = encoder.layer_norm(x, layer_base + ".norm_out");
    }

    mx::eval(x);
    save_f32(out_dir + "/encoder_output.f32", x);
    print_stats("encoder.output", x);

    std::cout << "\n=== Done. All activations saved to " << out_dir << " ===" << std::endl;
    std::cout << "Run compare script: python3 /tmp/compare_activations.py "
              << fixture_dir << " " << out_dir << std::endl;

    return 0;
}
