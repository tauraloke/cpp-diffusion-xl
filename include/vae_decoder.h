#pragma once

#include <string>
#include <memory>
#include <cuda_runtime.h>
#include <cudnn.h>

class VAEDecoder {
public:
    struct Config {
        std::string model_path;
        int latent_channels = 4;
        int latent_height = 128;  // For 1024x1024 output
        int latent_width = 128;
        int output_channels = 3;
        int output_height = 1024;
        int output_width = 1024;
        float scaling_factor = 0.18215f;
    };

    VAEDecoder();
    ~VAEDecoder();

    bool load_model(const std::string& model_path);
    void cleanup();
    
    // Decode latents to image
    bool decode(const float* latents, float* output, int batch_size = 1);
    
    // Model info
    bool is_loaded() const;
    int get_latent_channels() const;
    int get_latent_height() const;
    int get_latent_width() const;
    int get_output_channels() const;
    int get_output_height() const;
    int get_output_width() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};
