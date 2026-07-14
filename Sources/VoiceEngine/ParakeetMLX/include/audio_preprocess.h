// audio_preprocess.h — Mel spectrogram computation for parakeet.
#pragma once

#include "mlx/mlx.h"
#include "parakeet_model.h"

namespace parakeet {

namespace mx = mlx::core;

// Compute log-mel spectrogram from raw audio samples.
// Input: 1D float array of audio samples at sample_rate
// weights_dir: path containing hanning_window.f32 and mel_filterbank.f32
// Output: [1, n_mels, time_frames] float array
mx::array compute_log_mel(
    const mx::array& audio,
    const PreprocessorConfig& cfg,
    const std::string& weights_dir);

// Load audio from WAV file, resample to target sample rate.
// Returns 1D float32 array.
mx::array load_wav(const std::string& path, int target_sample_rate);

} // namespace parakeet
