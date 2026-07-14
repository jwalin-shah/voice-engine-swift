// weight_loader.cpp — Load flat .f32 weight files into MLX arrays.
// Uses auto-generated weight_entries.h (no JSON dependency).
#include "weight_loader.h"
#include "weights/weight_entries.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace parakeet {

std::unordered_map<std::string, mx::array> WeightLoader::load(
    const std::string& weights_dir) {

    std::unordered_map<std::string, mx::array> weights;

    for (const auto& entry : kWeightEntries) {
        std::string fpath = weights_dir + "/" + entry.filename;
        mx::array arr = load_f32(fpath, entry.shape);
        weights.insert({entry.key, arr});
    }

    std::cout << "[WeightLoader] Loaded " << kNumWeights
              << " weight tensors from " << weights_dir << std::endl;
    return weights;
}

mx::array WeightLoader::load_f32(
    const std::string& path, const std::vector<int>& shape) {

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open weight file: " + path);
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Create MLX array from raw data
    // We need to read into a float buffer first, then create the MLX array
    size_t num_elements = file_size / sizeof(float);
    std::vector<float> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);

    // Create MLX array with explicit shape
    auto arr = mx::array(data.data(), mx::Shape(shape.begin(), shape.end()), mx::float32);

    // Evaluate to force materialization
    mx::eval(arr);

    return arr;
}

std::string WeightLoader::to_manifest_key(const std::string& mlx_key) {
    std::string result = mlx_key;
    for (char& c : result) {
        if (c == '.') c = '_';
    }
    return result;
}

} // namespace parakeet
