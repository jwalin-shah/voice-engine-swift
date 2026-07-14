// tokenizer.cpp — BPE tokenizer for parakeet.
#include "tokenizer.h"

#include <fstream>
#include <iostream>
#include <algorithm>

namespace parakeet {

void Tokenizer::load(const std::string& path) {
    vocabulary_.clear();
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open vocab file: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Remove trailing \r if present
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        vocabulary_.push_back(line);
    }
    std::cout << "[Tokenizer] Loaded " << vocabulary_.size() << " tokens from " << path << std::endl;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int id : tokens) {
        result += decode_token(id);
    }
    return result;
}

std::string Tokenizer::decode_token(int token_id) const {
    if (token_id < 0 || token_id >= static_cast<int>(vocabulary_.size())) {
        return "";
    }
    std::string token = vocabulary_[token_id];
    // Replace "▁" (U+2581) with space
    std::string space_marker = "\xe2\x96\x81"; // UTF-8 for U+2581
    std::string result;
    for (size_t i = 0; i < token.size(); ) {
        if (i + 3 <= token.size() &&
            static_cast<unsigned char>(token[i]) == 0xE2 &&
            static_cast<unsigned char>(token[i+1]) == 0x96 &&
            static_cast<unsigned char>(token[i+2]) == 0x81) {
            result += ' ';
            i += 3;
        } else {
            result += token[i];
            i++;
        }
    }
    return result;
}

} // namespace parakeet
