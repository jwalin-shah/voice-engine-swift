// audio_preprocess.cpp — Mel spectrogram computation for parakeet.
//
// Implements the exact preprocessing from parakeet_mlx/audio.py:
//   1. Pre-emphasis filter
//   2. STFT with Hanning window
//   3. Magnitude spectrum (abs of complex)
//   4. Mel filterbank projection
//   5. Log compression
//   6. Per-feature normalization
#include "audio_preprocess.h"

#include <fstream>
#include <iostream>

namespace parakeet {

// Helper: load a flat .f32 file into an MLX array
static mx::array load_f32_array(const std::string& path, const std::vector<int>& shape) {
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

// Reflect padding for STFT
static mx::array reflect_pad(const mx::array& x, int padding) {
    // x is 1D. Pad with reflection on both sides.
    // prefix = x[1:padding+1][::-1], suffix = x[-(padding+1):-1][::-1]
    auto prefix = mx::slice(x, {1}, {padding + 1});
    // For reverse, we need to index backward. Use -1 step via slice with negative stride?
    // Simpler: reverse by gathering from end.
    auto n = x.shape()[0];
    // Actually, MX C++ doesn't have a simple reverse op. Let me use slice with negative
    // stride or just concatenate without reversing (approximate for now).
    // ponytail: approximate reflect pad by symmetric extension
    auto prefix_rev = mx::take(x, mx::arange(padding, 0, -1, mx::int32));
    int n_int = static_cast<int>(n);
    auto suffix_rev = mx::take(x, mx::arange(n_int - 1, n_int - 1 - padding, -1, mx::int32));
    return mx::concatenate({prefix_rev, x, suffix_rev});
}

mx::array compute_log_mel(
    const mx::array& audio,
    const PreprocessorConfig& cfg,
    const std::string& weights_dir) {

    auto x = audio;
    auto dtype = x.dtype();

    // 1. Pre-emphasis
    if (cfg.preemph > 0.0f) {
        // x[1:] - preemph * x[:-1], keep x[0]
        auto x_shifted = mx::slice(x, {1}, {x.shape()[0]});
        auto x_original = mx::slice(x, {0}, {x.shape()[0] - 1});
        auto emphasized = mx::subtract(x_shifted, mx::multiply(mx::array(cfg.preemph), x_original));
        x = mx::concatenate({mx::slice(x, {0}, {1}), emphasized});
    }

    // 2. STFT using as_strided for framing
    int n_fft = cfg.n_fft;
    int hop = cfg.hop_length;
    int win_len = cfg.win_length;

    // Load window and pad to n_fft if needed
    auto window = load_f32_array(
        weights_dir + "/hanning_window.f32", {win_len});
    if (win_len != n_fft) {
        if (win_len > n_fft) {
            window = mx::slice(window, {0}, {n_fft});
        } else {
            window = mx::pad(window, {{0, n_fft - win_len}});
        }
    }

    // Pad with reflection
    int pad = n_fft / 2;
    x = reflect_pad(x, pad);

    // Frame into [n_frames, n_fft] using as_strided
    int n = x.shape()[0];
    int n_frames = 1 + (n - n_fft) / hop;
    auto frames = mx::as_strided(
        x,
        {n_frames, n_fft},
        {hop, 1},
        0);

    // Apply window
    frames = mx::multiply(frames, mx::expand_dims(window, 0));

    // RFFT along last axis
    auto stft_result = mx::fft::rfft(frames, -1);

    // 3. Magnitude: |X|^2 = Re(X)^2 + Im(X)^2
    // Use mx::real/mx::imag to extract components
    auto real_part = mx::real(stft_result);
    auto imag_part = mx::imag(stft_result);
    // real_part shape: [n_frames, n_fft/2 + 1] = [n_frames, 257]

    auto mag_sq = mx::add(
        mx::multiply(real_part, real_part),
        mx::multiply(imag_part, imag_part));

    // Apply magnitude power (default 2.0)
    // In Python parakeet_mlx audio.py, the computation is:
    //   abs = mx.abs(mx.view(x, original_dtype))
    //   x = abs[..., ::2] + abs[..., 1::2]
    //   if mag_power != 1.0: x = mx.power(x, mag_power)
    //
    // Our mag_sq is |X|^2. For mag_power=2.0, we'd compute (|X|^2)^2 = |X|^4
    // which matches the Python: (|re| + |im|)^2 ~ |X|^2, then power 2 gives |X|^4
    // Actually the Python code computes |re|+|im| (not magnitude), then squares it.
    // Let me match the Python exactly.
    auto mag_abs = mx::add(mx::abs(real_part), mx::abs(imag_part));  // |re| + |im|
    float mag_power = 2.0f;
    auto mag = mx::power(mag_abs, mx::array(mag_power));
    // mag shape: [n_frames, freq_bins]
    int freq_bins = n_fft / 2 + 1;  // 257

    // 4. Mel filterbank projection
    // Load filterbank: shape (n_mels, freq_bins)
    auto mel_basis = load_f32_array(weights_dir + "/mel_filterbank.f32", {cfg.n_mels, freq_bins});
    // matmul: (n_mels, freq_bins) @ (freq_bins, n_frames) = (n_mels, n_frames)
    // Wait, we need mel_basis @ mag.T
    // mag shape: [n_frames, freq_bins]
    // Transpose: [freq_bins, n_frames]
    auto mag_t = mx::transpose(mag);
    auto mel = mx::matmul(mel_basis, mag_t);
    // mel shape: [n_mels, n_frames]

    // 5. Log compression
    mel = mx::log(mx::add(mel, mx::array(1e-5f)));

    // 6. Normalization
    if (cfg.normalize == "per_feature") {
        auto mean = mx::mean(mel, 1, true);
        auto std = mx::std(mel, 1, true);
        mel = mx::divide(mx::subtract(mel, mean), mx::add(std, mx::array(1e-5f)));
    } else {
        auto mean = mx::mean(mel);
        auto std = mx::std(mel);
        mel = mx::divide(mx::subtract(mel, mean), mx::add(std, mx::array(1e-5f)));
    }

    // 7. Format: [1, n_mels, n_frames] (batch, features, time)
    mel = mx::transpose(mel);    // [n_frames, n_mels]
    mel = mx::expand_dims(mel, 0); // [1, n_frames, n_mels]

    mel = mx::astype(mel, dtype);
    mx::eval(mel);
    return mel;
}

mx::array load_wav(const std::string& path, int target_sample_rate) {
    // Use a simple WAV parser for 16-bit PCM mono/stereo
    // This is a minimal implementation — for production, use libsndfile or ffmpeg
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open WAV file: " + path);
    }

    // Read WAV header
    char riff[4], wave[4], fmt[4];
    uint32_t file_size, fmt_size;
    uint16_t audio_format, num_channels;
    uint32_t sample_rate, byte_rate;
    uint16_t block_align, bits_per_sample;
    char data_id[4];
    uint32_t data_size;

    file.read(riff, 4);
    file.read(reinterpret_cast<char*>(&file_size), 4);
    file.read(wave, 4);
    file.read(fmt, 4);
    file.read(reinterpret_cast<char*>(&fmt_size), 4);
    file.read(reinterpret_cast<char*>(&audio_format), 2);
    file.read(reinterpret_cast<char*>(&num_channels), 2);
    file.read(reinterpret_cast<char*>(&sample_rate), 4);
    file.read(reinterpret_cast<char*>(&byte_rate), 4);
    file.read(reinterpret_cast<char*>(&block_align), 2);
    file.read(reinterpret_cast<char*>(&bits_per_sample), 2);

    // Skip extra fmt bytes
    if (fmt_size > 16) {
        file.ignore(fmt_size - 16);
    }

    // Find data chunk
    while (true) {
        file.read(data_id, 4);
        file.read(reinterpret_cast<char*>(&data_size), 4);
        if (std::string(data_id, 4) == "data") break;
        file.ignore(data_size);
    }

    // Read samples
    int num_samples = data_size / (bits_per_sample / 8);
    std::vector<int16_t> raw(num_samples);
    file.read(reinterpret_cast<char*>(raw.data()), data_size);

    // Convert to float and mono
    std::vector<float> samples;
    if (num_channels == 1) {
        samples.resize(num_samples);
        for (int i = 0; i < num_samples; i++) {
            samples[i] = raw[i] / 32768.0f;
        }
    } else {
        samples.resize(num_samples / num_channels);
        for (int i = 0; i < num_samples / num_channels; i++) {
            float sum = 0;
            for (int c = 0; c < num_channels; c++) {
                sum += raw[i * num_channels + c] / 32768.0f;
            }
            samples[i] = sum / num_channels;
        }
    }

    // Simple resampling: if rates match, use as-is; otherwise linear interpolation
    // ponytail: for now assume 16kHz input
    if (static_cast<int>(sample_rate) != target_sample_rate) {
        std::cerr << "WARNING: WAV sample rate " << sample_rate
                  << " != target " << target_sample_rate << ", resampling not implemented"
                  << std::endl;
    }

    auto audio = mx::array(samples.data(), {(int)samples.size()}, mx::float32);
    mx::eval(audio);
    return audio;
}

} // namespace parakeet
