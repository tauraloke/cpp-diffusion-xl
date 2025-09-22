#pragma once

#include <string>
#include <memory>
#include <cuda_runtime.h>
#include <cudnn.h>

class UNetModel {
public:
struct Config {
        std::string model_path;
        int in_channels = 4;
        int out_channels = 4;
        int model_channels = 320;
        int attention_resolutions[3] = {4, 2, 1};
        int num_res_blocks = 2;
        int channel_mult[4] = {1, 2, 4, 4};
        int num_heads = 8;
        int num_head_channels = 64;
        int num_heads_upsample = -1;
        bool use_scale_shift_norm = true;
        int dropout = 0;
        int context_dim = 2048;  // SDXL uses 2048 for text embedding
        bool use_spatial_transformer = true;
        std::string use_linear_in_transformer = "False";
        int adm_in_channels = 2816;  // SDXL specific
    };

    UNetModel();
    ~UNetModel();

    bool load_model(const std::string& model_path);
    void cleanup();
    
    // Forward pass
    bool forward(
        const float* x,           // Input latents
        const float* timestep,    // Timestep
        const float* context,     // Text embeddings
        const float* y,           // Additional conditioning (optional)
        float* output,            // Output noise prediction
        int batch_size = 1
    );
    
    // Model info
    bool is_loaded() const;
    int get_in_channels() const;
    int get_out_channels() const;
    int get_context_dim() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};
