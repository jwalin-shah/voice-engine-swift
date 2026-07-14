// weight_loader.h — Load flat .f32 weight files into MLX arrays.
#pragma once

#include "mlx/mlx.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace parakeet {

namespace mx = mlx::core;

class WeightLoader {
public:
    // Load weights from a directory containing .f32 files and weight_manifest.json.
    // Returns a map from weight name to MLX array.
    static std::unordered_map<std::string, mx::array> load(
        const std::string& weights_dir);

    // Load a single .f32 file into an MLX array with the given shape.
    static mx::array load_f32(const std::string& path, const std::vector<int>& shape);

private:
    // Convert MLX weight key to our manifest key (dots -> underscores)
    static std::string to_manifest_key(const std::string& mlx_key);
};

} // namespace parakeet
