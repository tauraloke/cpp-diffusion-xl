#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>

class Sampler {
public:
    enum class Type {
        EULER_A,
        EULER,
        HEUN,
        DPM_2,
        DPM_2_ANCESTRAL,
        LMS,
        DPM_FAST,
        DPM_ADAPTIVE,
        DPMPP_2S_ANCESTRAL,
        DPMPP_SDE,
        DPMPP_2M,
        DPMPP_2M_SDE,
        DDPM,
        LCM
    };

    struct Config {
        Type type = Type::EULER_A;
        int steps = 20;
        float eta = 1.0f;
        float s_churn = 0.0f;
        float s_tmin = 0.0f;
        float s_tmax = 1e10f;
        float s_noise = 1.0f;
        unsigned long seed = 0;
    };

    Sampler();
    ~Sampler();

    bool initialize();
    void cleanup();
    
    // Main sampling function
    bool sample(
        void* model,  // UNet model
        float* noise,  // Input noise tensor
        float* sigmas, // Noise schedule
        int batch_size,
        int channels,
        int height,
        int width,
        const Config& config,
        void* callback = nullptr
    );

    // Sampler type conversion
    static Type string_to_type(const std::string& name);
    static std::string type_to_string(Type type);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};
