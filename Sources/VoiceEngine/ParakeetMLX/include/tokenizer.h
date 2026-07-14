// tokenizer.h — BPE tokenizer for parakeet.
#pragma once

#include <string>
#include <vector>

namespace parakeet {

class Tokenizer {
public:
    // Load vocabulary from vocab.json
    void load(const std::string& path);

    // Decode token IDs to text
    // The vocabulary uses "▁" (U+2581) as space marker, which we replace with " "
    std::string decode(const std::vector<int>& tokens) const;

    // Decode a single token
    std::string decode_token(int token_id) const;

    int vocab_size() const { return static_cast<int>(vocabulary_.size()); }

private:
    std::vector<std::string> vocabulary_;
};

} // namespace parakeet
