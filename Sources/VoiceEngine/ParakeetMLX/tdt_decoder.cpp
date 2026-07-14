// tdt_decoder.cpp — TDT decoder implementation (map-based weights).
#include "tdt_decoder.h"
#include <cmath>
#include <iostream>

namespace parakeet {

TDTDecoder::TDTDecoder(
    const DecoderConfig& dec_cfg,
    const JointConfig& joint_cfg,
    const DecodingConfig& decoding_cfg)
    : dec_cfg_(dec_cfg), joint_cfg_(joint_cfg), decoding_cfg_(decoding_cfg) {
    std::cout << "[TDTDecoder] vocab=" << dec_cfg_.vocab_size
              << ", pred_hidden=" << dec_cfg_.pred_hidden
              << ", joint_hidden=" << joint_cfg_.joint_hidden << std::endl;
}

void TDTDecoder::load_weights(
    const std::unordered_map<std::string, mx::array>& weights) {
    w_ = weights;
    std::cout << "[TDTDecoder] Weights loaded" << std::endl;
}

std::pair<mx::array, std::pair<mx::array, mx::array>>
TDTDecoder::lstm_step(
    const mx::array& x,
    const mx::array& Wh, const mx::array& Wx, const mx::array& bias,
    const mx::array& h_prev, const mx::array& c_prev) const {

    int hidden = h_prev.shape()[1];
    auto gates = mx::add(
        mx::add(
            mx::matmul(x, mx::transpose(Wx)),
            mx::matmul(h_prev, mx::transpose(Wh))),
        bias);

    int B = h_prev.shape()[0];  // batch size for gates shape
    auto i_gate = mx::sigmoid(mx::slice(gates, {0, 0}, {B, hidden}));
    auto f_gate = mx::sigmoid(mx::slice(gates, {0, hidden}, {B, 2 * hidden}));
    auto g_gate = mx::tanh(mx::slice(gates, {0, 2 * hidden}, {B, 3 * hidden}));
    auto o_gate = mx::sigmoid(mx::slice(gates, {0, 3 * hidden}, {B, 4 * hidden}));

    auto c_new = mx::add(
        mx::multiply(f_gate, c_prev),
        mx::multiply(i_gate, g_gate));
    auto h_new = mx::multiply(o_gate, mx::tanh(c_new));

    return {h_new, {h_new, c_new}};
}

TDTDecoder::PredictResult TDTDecoder::predict(
    int token,
    const mx::array& h0, const mx::array& c0,
    const mx::array& h1, const mx::array& c1) const {

    int pred_hidden = dec_cfg_.pred_hidden;
    auto embed_w = W("decoder.prediction.embed.weight");
    auto emb = mx::take(embed_w, token, 0);
    emb = mx::reshape(emb, {1, pred_hidden});

    auto lstm0_Wh = W("decoder.prediction.dec_rnn.lstm.0.Wh");
    auto lstm0_Wx = W("decoder.prediction.dec_rnn.lstm.0.Wx");
    auto lstm0_b  = W("decoder.prediction.dec_rnn.lstm.0.bias");
    auto [h0_out, h0_state] = lstm_step(emb, lstm0_Wh, lstm0_Wx, lstm0_b, h0, c0);

    auto lstm1_Wh = W("decoder.prediction.dec_rnn.lstm.1.Wh");
    auto lstm1_Wx = W("decoder.prediction.dec_rnn.lstm.1.Wx");
    auto lstm1_b  = W("decoder.prediction.dec_rnn.lstm.1.bias");
    auto [h1_out, h1_state] = lstm_step(h0_out, lstm1_Wh, lstm1_Wx, lstm1_b, h0_state.first, h0_state.second);

    return {
        mx::expand_dims(h1_out, 1),
        h0_state.first, h0_state.second,
        h1_state.first, h1_state.second
    };
}

TDTDecoder::PredictResult TDTDecoder::predict_zero(
    const mx::array& h0, const mx::array& c0,
    const mx::array& h1, const mx::array& c1) const {

    int pred_hidden = dec_cfg_.pred_hidden;
    auto zero_input = mx::zeros({1, pred_hidden}, mx::float32);

    auto lstm0_Wh = W("decoder.prediction.dec_rnn.lstm.0.Wh");
    auto lstm0_Wx = W("decoder.prediction.dec_rnn.lstm.0.Wx");
    auto lstm0_b  = W("decoder.prediction.dec_rnn.lstm.0.bias");
    auto [h0_out, h0_state] = lstm_step(zero_input, lstm0_Wh, lstm0_Wx, lstm0_b, h0, c0);

    auto lstm1_Wh = W("decoder.prediction.dec_rnn.lstm.1.Wh");
    auto lstm1_Wx = W("decoder.prediction.dec_rnn.lstm.1.Wx");
    auto lstm1_b  = W("decoder.prediction.dec_rnn.lstm.1.bias");
    auto [h1_out, h1_state] = lstm_step(h0_out, lstm1_Wh, lstm1_Wx, lstm1_b, h0_state.first, h0_state.second);

    return {
        mx::expand_dims(h1_out, 1),
        h0_state.first, h0_state.second,
        h1_state.first, h1_state.second
    };
}

TDTDecoder::BatchedPredictResult TDTDecoder::predict_batched(
    const std::vector<int>& tokens,
    const mx::array& h0, const mx::array& c0,
    const mx::array& h1, const mx::array& c1) const {

    int B = static_cast<int>(tokens.size());
    int pred_hidden = dec_cfg_.pred_hidden;

    // Embed all tokens at once: [B] -> [B, pred_hidden]
    auto tok_arr = mx::array(tokens.data(), {B}, mx::int32);
    auto embed_w = W("decoder.prediction.embed.weight");
    auto emb = mx::take(embed_w, tok_arr, 0);  // [B, pred_hidden]

    auto lstm0_Wh = W("decoder.prediction.dec_rnn.lstm.0.Wh");
    auto lstm0_Wx = W("decoder.prediction.dec_rnn.lstm.0.Wx");
    auto lstm0_b  = W("decoder.prediction.dec_rnn.lstm.0.bias");
    auto [h0_out, h0_state] = lstm_step(emb, lstm0_Wh, lstm0_Wx, lstm0_b, h0, c0);

    auto lstm1_Wh = W("decoder.prediction.dec_rnn.lstm.1.Wh");
    auto lstm1_Wx = W("decoder.prediction.dec_rnn.lstm.1.Wx");
    auto lstm1_b  = W("decoder.prediction.dec_rnn.lstm.1.bias");
    auto [h1_out, h1_state] = lstm_step(h0_out, lstm1_Wh, lstm1_Wx, lstm1_b, h0_state.first, h0_state.second);

    // h1_out is [B, pred_hidden], expand dims to [B, 1, pred_hidden]
    return {
        mx::expand_dims(h1_out, 1),
        h0_state.first, h0_state.second,
        h1_state.first, h1_state.second
    };
}

mx::array TDTDecoder::joint(
    const mx::array& enc_step,
    const mx::array& pred_out) const {

    auto joint_enc_w = W("joint.enc.weight");
    auto joint_enc_b = W("joint.enc.bias");
    auto joint_pred_w = W("joint.pred.weight");
    auto joint_pred_b = W("joint.pred.bias");
    auto joint_out_w = W("joint.joint_net.2.weight");
    auto joint_out_b = W("joint.joint_net.2.bias");

    auto enc = mx::add(mx::matmul(enc_step, mx::transpose(joint_enc_w)), joint_enc_b);
    auto pred = mx::add(mx::matmul(pred_out, mx::transpose(joint_pred_w)), joint_pred_b);
    auto combined = mx::add(mx::expand_dims(enc, 2), mx::expand_dims(pred, 1));
    combined = mx::maximum(combined, mx::array(0.0f));  // relu
    auto logits = mx::add(mx::matmul(combined, mx::transpose(joint_out_w)), joint_out_b);
    return mx::reshape(logits, {1, 1, -1});
}

std::vector<std::vector<int>> TDTDecoder::decode(
    const mx::array& encoder_features,
    const mx::array& encoder_lengths) {

    int B = encoder_features.shape()[0];
    int T_enc = encoder_features.shape()[1];
    int vocab_size = dec_cfg_.vocab_size;
    int blank_id = vocab_size;
    int pred_hidden = dec_cfg_.pred_hidden;
    const auto& durations = decoding_cfg_.durations;

    mx::eval(encoder_features, encoder_lengths);

    std::vector<std::vector<int>> results(B);

    for (int b = 0; b < B; b++) {
        std::vector<int> hypothesis;
        int length = T_enc;
        // ponytail: assume length = T_enc (full feature sequence length)

        auto feature = mx::slice(encoder_features, {b, 0, 0}, {b + 1, T_enc, encoder_features.shape()[2]});

        auto h0 = mx::zeros({1, pred_hidden}, mx::float32);
        auto c0 = mx::zeros({1, pred_hidden}, mx::float32);
        auto h1 = mx::zeros({1, pred_hidden}, mx::float32);
        auto c1 = mx::zeros({1, pred_hidden}, mx::float32);

        int step = 0, last_token = -1, new_symbols = 0;

        while (step < length) {
            PredictResult pr = [&]() -> PredictResult {
                if (last_token >= 0) {
                    return predict(last_token, h0, c0, h1, c1);
                }
                // First step: zero input through LSTMs
                return predict_zero(h0, c0, h1, c1);
            }();

            auto enc_step = mx::slice(feature, {0, step, 0}, {1, step + 1, feature.shape()[2]});
            auto joint_out = joint(enc_step, pr.output);
            mx::eval(joint_out);

            auto token_logits = mx::slice(joint_out, {0, 0, 0}, {1, 1, vocab_size + 1});
            auto token_flat = mx::reshape(token_logits, {vocab_size + 1});
            mx::eval(token_flat);
            int pred_token = static_cast<int>(mx::argmax(token_flat, 0, false).item<uint32_t>());

            auto dur_logits = mx::slice(joint_out, {0, 0, vocab_size + 1}, {1, 1, joint_out.shape()[2]});
            auto dur_flat = mx::reshape(dur_logits, {static_cast<int>(durations.size())});
            mx::eval(dur_flat);
            int decision = static_cast<int>(mx::argmax(dur_flat, 0, false).item<uint32_t>());

            if (pred_token != blank_id) {
                hypothesis.push_back(pred_token);
                last_token = pred_token;
                h0 = pr.h0; c0 = pr.c0;
                h1 = pr.h1; c1 = pr.c1;
            }

            int dur = (decision < static_cast<int>(durations.size())) ? durations[decision] : 0;
            step += dur;
            new_symbols++;
            if (dur != 0) {
                new_symbols = 0;
            } else if (new_symbols >= decoding_cfg_.max_symbols) {
                step += 1;
                new_symbols = 0;
            }
        }

        results[b] = hypothesis;
    }

    return results;
}

std::vector<std::vector<int>> TDTDecoder::decode_beam(
    const mx::array& encoder_features,
    const mx::array& encoder_lengths,
    int beam_width) {

    int B = encoder_features.shape()[0];
    int T_enc = encoder_features.shape()[1];
    int vocab_size = dec_cfg_.vocab_size;
    int blank_id = vocab_size;
    int pred_hidden = dec_cfg_.pred_hidden;
    const auto& durations = decoding_cfg_.durations;
    int num_durs = static_cast<int>(durations.size());
    int num_classes = vocab_size + 1 + num_durs;

    mx::eval(encoder_features, encoder_lengths);

    std::vector<std::vector<int>> results(B);

    for (int b = 0; b < B; b++) {
        auto feature = mx::slice(encoder_features, {b, 0, 0}, {b + 1, T_enc, encoder_features.shape()[2]});

        // Beam state stored as float vectors (not mx::array) for cheap copy
        struct Beam {
            std::vector<int> tokens;
            int pos = 0;
            int last_token = -1;
            int stuck = 0;
            float score = 0.0f;
            std::vector<float> h0d, c0d, h1d, c1d;
        };

        Beam init_beam;
        init_beam.pos = 0;
        init_beam.last_token = -1;
        init_beam.score = 0.0f;
        {
            auto zh0 = mx::zeros({1, pred_hidden}, mx::float32);
            auto zc0 = mx::zeros({1, pred_hidden}, mx::float32);
            auto zh1 = mx::zeros({1, pred_hidden}, mx::float32);
            auto zc1 = mx::zeros({1, pred_hidden}, mx::float32);
            mx::eval(zh0, zc0, zh1, zc1);
            init_beam.h0d.assign(zh0.data<float>(), zh0.data<float>() + pred_hidden);
            init_beam.c0d.assign(zc0.data<float>(), zc0.data<float>() + pred_hidden);
            init_beam.h1d.assign(zh1.data<float>(), zh1.data<float>() + pred_hidden);
            init_beam.c1d.assign(zc1.data<float>(), zc1.data<float>() + pred_hidden);
        }

        std::vector<Beam> beams = {init_beam};

        // — helper: stack beam state float vectors into [N, pred_hidden] mx::arrays —
        auto stack_states = [&](const std::vector<Beam*>& beam_ptrs) {
            int N = static_cast<int>(beam_ptrs.size());
            std::vector<float> buf_h0(N * pred_hidden), buf_c0(N * pred_hidden);
            std::vector<float> buf_h1(N * pred_hidden), buf_c1(N * pred_hidden);
            for (int i = 0; i < N; i++) {
                std::memcpy(buf_h0.data() + i * pred_hidden, beam_ptrs[i]->h0d.data(), pred_hidden * sizeof(float));
                std::memcpy(buf_c0.data() + i * pred_hidden, beam_ptrs[i]->c0d.data(), pred_hidden * sizeof(float));
                std::memcpy(buf_h1.data() + i * pred_hidden, beam_ptrs[i]->h1d.data(), pred_hidden * sizeof(float));
                std::memcpy(buf_c1.data() + i * pred_hidden, beam_ptrs[i]->c1d.data(), pred_hidden * sizeof(float));
            }
            auto h0a = mx::array(buf_h0.data(), {N, pred_hidden}, mx::float32);
            auto c0a = mx::array(buf_c0.data(), {N, pred_hidden}, mx::float32);
            auto h1a = mx::array(buf_h1.data(), {N, pred_hidden}, mx::float32);
            auto c1a = mx::array(buf_c1.data(), {N, pred_hidden}, mx::float32);
            return std::make_tuple(h0a, c0a, h1a, c1a);
        };

        int enc_dim = feature.shape()[2];
        mx::eval(feature);  // materialize once (feature is a slice of already-evaluated parent)
        const float* feat_data = feature.data<float>();

        // log_softmax inline (avoids mx::array overhead per beam)
        auto log_softmax_vec = [](const float* v, int n, std::vector<float>& out) {
            out.resize(n);
            float max_v = *std::max_element(v, v + n);
            float sum_exp = 0.0f;
            for (int j = 0; j < n; j++) sum_exp += std::exp(v[j] - max_v);
            float log_sum = std::log(sum_exp);
            for (int j = 0; j < n; j++) out[j] = (v[j] - max_v) - log_sum;
        };

        int max_steps = T_enc * 3;  // generous bound with stuck guard
        int steps = 0;

        while (!beams.empty() && steps < max_steps) {
            steps++;

            // — 1. Separate finished beams (pass through unchanged) from active beams —
            std::vector<Beam> finished_beams;
            std::vector<Beam> active_beams;
            for (auto& bm : beams) {
                if (bm.pos >= T_enc)
                    finished_beams.push_back(std::move(bm));
                else
                    active_beams.push_back(std::move(bm));
            }

            if (active_beams.empty()) {
                beams = std::move(finished_beams);
                break;
            }

            int num_active = static_cast<int>(active_beams.size());

            // — 2. Classify active beams: normal (last_token >= 0) vs initial (last_token < 0) —
            std::vector<int> normal_idx, initial_idx;
            std::vector<int> normal_tokens;
            for (int i = 0; i < num_active; i++) {
                if (active_beams[i].last_token >= 0) {
                    normal_idx.push_back(i);
                    normal_tokens.push_back(active_beams[i].last_token);
                } else {
                    initial_idx.push_back(i);
                }
            }

            // — 3. Predictor: batched for normal beams, individual for initial beams —
            //     ponytail: initial beams are rare (at most 1), so individual calls are ok
            //     Build pred_stacked buffer and per-beam new states directly (avoids vector<mx::array>)

            std::vector<float> pred_buf(num_active * pred_hidden);
            std::vector<float> new_h0_buf(num_active * pred_hidden, 0.0f);
            std::vector<float> new_c0_buf(num_active * pred_hidden, 0.0f);
            std::vector<float> new_h1_buf(num_active * pred_hidden, 0.0f);
            std::vector<float> new_c1_buf(num_active * pred_hidden, 0.0f);

            if (!normal_idx.empty()) {
                int N = static_cast<int>(normal_idx.size());
                std::vector<Beam*> normal_ptrs;
                for (int idx : normal_idx) normal_ptrs.push_back(&active_beams[idx]);
                auto [h0a, c0a, h1a, c1a] = stack_states(normal_ptrs);

                auto batched = predict_batched(normal_tokens, h0a, c0a, h1a, c1a);
                mx::eval(batched.output, batched.h0, batched.c0, batched.h1, batched.c1);

                const float* out_ptr = batched.output.data<float>();  // [N, 1, pred_hidden] row-major
                const float* h0_ptr = batched.h0.data<float>();
                const float* c0_ptr = batched.c0.data<float>();
                const float* h1_ptr = batched.h1.data<float>();
                const float* c1_ptr = batched.c1.data<float>();

                for (int j = 0; j < N; j++) {
                    int bi = normal_idx[j];
                    std::memcpy(pred_buf.data() + bi * pred_hidden, out_ptr + j * pred_hidden, pred_hidden * sizeof(float));
                    std::memcpy(new_h0_buf.data() + bi * pred_hidden, h0_ptr + j * pred_hidden, pred_hidden * sizeof(float));
                    std::memcpy(new_c0_buf.data() + bi * pred_hidden, c0_ptr + j * pred_hidden, pred_hidden * sizeof(float));
                    std::memcpy(new_h1_buf.data() + bi * pred_hidden, h1_ptr + j * pred_hidden, pred_hidden * sizeof(float));
                    std::memcpy(new_c1_buf.data() + bi * pred_hidden, c1_ptr + j * pred_hidden, pred_hidden * sizeof(float));
                }
            }

            for (int bi : initial_idx) {
                const auto& bm = active_beams[bi];
                auto h0a = mx::array(bm.h0d.data(), {1, pred_hidden}, mx::float32);
                auto c0a = mx::array(bm.c0d.data(), {1, pred_hidden}, mx::float32);
                auto h1a = mx::array(bm.h1d.data(), {1, pred_hidden}, mx::float32);
                auto c1a = mx::array(bm.c1d.data(), {1, pred_hidden}, mx::float32);

                auto pr = predict_zero(h0a, c0a, h1a, c1a);
                mx::eval(pr.output, pr.h0, pr.c0, pr.h1, pr.c1);

                std::memcpy(pred_buf.data() + bi * pred_hidden, pr.output.data<float>(), pred_hidden * sizeof(float));
                std::memcpy(new_h0_buf.data() + bi * pred_hidden, pr.h0.data<float>(), pred_hidden * sizeof(float));
                std::memcpy(new_c0_buf.data() + bi * pred_hidden, pr.c0.data<float>(), pred_hidden * sizeof(float));
                std::memcpy(new_h1_buf.data() + bi * pred_hidden, pr.h1.data<float>(), pred_hidden * sizeof(float));
                std::memcpy(new_c1_buf.data() + bi * pred_hidden, pr.c1.data<float>(), pred_hidden * sizeof(float));
            }

            // — 4. Gather encoder frames + stack predictor outputs → single batched joint —
            //     ponytail: gather on CPU; beam_width is small, MLX gather overhead dominates
            std::vector<float> enc_buf(num_active * enc_dim);
            for (int i = 0; i < num_active; i++) {
                int pos = active_beams[i].pos;
                std::memcpy(enc_buf.data() + i * enc_dim,
                           feat_data + pos * enc_dim,
                           enc_dim * sizeof(float));
            }
            auto enc_stacked = mx::array(enc_buf.data(), {num_active, 1, enc_dim}, mx::float32);
            auto pred_stacked = mx::array(pred_buf.data(), {num_active, 1, pred_hidden}, mx::float32);

            auto joint_out = joint(enc_stacked, pred_stacked);
            mx::eval(joint_out);

            // Reshape to [num_active, num_classes] for row-wise access
            auto joint_2d = mx::reshape(joint_out, {num_active, num_classes});
            mx::eval(joint_2d);
            const float* joint_ptr = joint_2d.data<float>();

            // — 5. Candidate expansion —
            int top_k = std::min(beam_width, vocab_size + 1);
            int top_d = std::min(beam_width, num_durs);

            std::vector<Beam> next_beams = std::move(finished_beams);

            for (int i = 0; i < num_active; i++) {
                const auto& bm = active_beams[i];
                const float* tok_base = joint_ptr + i * num_classes;
                const float* dur_base = tok_base + vocab_size + 1;

                std::vector<float> token_logprobs, dur_logprobs;
                log_softmax_vec(tok_base, vocab_size + 1, token_logprobs);
                log_softmax_vec(dur_base, num_durs, dur_logprobs);

                // Top-K tokens
                std::vector<std::pair<float, int>> tok_cands;
                for (int t = 0; t < vocab_size + 1; t++)
                    tok_cands.push_back({token_logprobs[t], t});
                std::partial_sort(tok_cands.begin(), tok_cands.begin() + top_k, tok_cands.end(),
                    std::greater<>());
                tok_cands.resize(top_k);

                // Top-K durations
                std::vector<std::pair<float, int>> dur_cands;
                for (int d = 0; d < num_durs; d++)
                    dur_cands.push_back({dur_logprobs[d], d});
                std::partial_sort(dur_cands.begin(), dur_cands.begin() + top_d, dur_cands.end(),
                    std::greater<>());
                dur_cands.resize(top_d);

                for (const auto& [tok_lp, token] : tok_cands) {
                    bool is_blank = (token == blank_id);
                    for (const auto& [dur_lp, dur_idx] : dur_cands) {
                        Beam cand = bm;
                        cand.score += tok_lp + dur_lp;

                        // Stuck guard: if duration=0 repeated max_symbols times, force advance
                        int dur = durations[dur_idx];
                        int new_stuck = (dur == 0) ? bm.stuck + 1 : 0;
                        if (new_stuck >= decoding_cfg_.max_symbols) {
                            cand.pos = bm.pos + 1;
                            cand.stuck = 0;
                        } else {
                            cand.pos = bm.pos + dur;
                            cand.stuck = new_stuck;
                        }

                        if (!is_blank) {
                            cand.tokens.push_back(token);
                            cand.last_token = token;
                            // Copy new LSTM states from float buffers (already evaluated above)
                            auto cp_row = [&](std::vector<float>& d, const float* src) {
                                d.assign(src, src + pred_hidden);
                            };
                            cp_row(cand.h0d, new_h0_buf.data() + i * pred_hidden);
                            cp_row(cand.c0d, new_c0_buf.data() + i * pred_hidden);
                            cp_row(cand.h1d, new_h1_buf.data() + i * pred_hidden);
                            cp_row(cand.c1d, new_c1_buf.data() + i * pred_hidden);
                        }
                        // blank: keep old LSTM states and last_token (already copied via bm)

                        next_beams.push_back(std::move(cand));
                    }
                }
            }

            // — 6. Prune to beam_width —
            if (next_beams.size() > static_cast<size_t>(beam_width)) {
                std::partial_sort(next_beams.begin(), next_beams.begin() + beam_width,
                    next_beams.end(),
                    [](const Beam& a, const Beam& b) { return a.score > b.score; });
                next_beams.resize(beam_width);
            }

            beams = std::move(next_beams);
        }

        // Pick best beam (highest score)
        if (!beams.empty()) {
            std::sort(beams.begin(), beams.end(),
                [](const Beam& a, const Beam& b) { return a.score > b.score; });
            results[b] = beams[0].tokens;
        }
    }

    return results;
}

} // namespace parakeet
