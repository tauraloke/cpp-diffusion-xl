#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cudnn.h>

class CLIPEncoder {
public:
    struct Config {
        std::string model_path;
        int max_length = 77;
        int embedding_size = 1280;
        bool freeze = true;
        std::string layer = "penultimate";
        int layer_idx = -2;
    };

    CLIPEncoder();
    ~CLIPEncoder();

    bool load_model(const std::string& model_path);
    void cleanup();
    
    // Text encoding
    bool encode_text(const std::string& text, float* output, int batch_size = 1);
    bool encode_text_batch(const std::vector<std::string>& texts, float* output);
    
    // Tokenization
    std::vector<int> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int>& tokens);
    
    // Model info
    int get_max_length() const;
    int get_embedding_size() const;
    bool is_loaded() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};
