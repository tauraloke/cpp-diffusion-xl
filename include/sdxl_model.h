#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cudnn.h>

class SDXLModel {
public:
    struct Config {
        std::string model_path;
        int width = 1024;
        int height = 1024;
        int steps = 20;
        float cfg_scale = 7.0f;
        std::string sampler = "euler_a";
        std::string scheduler = "sgm_uniform";
        std::string positive_prompt;
        std::string negative_prompt;
        unsigned long seed = 0;
    };

    SDXLModel();
    ~SDXLModel();

    bool load_model(const std::string& model_path);
    bool generate_image(const Config& config, const std::string& output_path);
    
    // Memory management
    void set_memory_fraction(float fraction);
    void enable_mixed_precision(bool enable);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

