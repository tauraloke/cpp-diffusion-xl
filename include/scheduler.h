#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>

class Scheduler {
public:
    enum class Type {
        NORMAL,
        KARRAS,
        EXPONENTIAL,
        SGM_UNIFORM,
        SIMPLE,
        DDIM_UNIFORM,
        BETA,
        LINEAR_QUADRATIC,
        KL_OPTIMAL
    };

    struct Config {
        Type type = Type::SGM_UNIFORM;
        int steps = 20;
        float sigma_min = 0.0292f;
        float sigma_max = 14.6146f;
        float rho = 7.0f;
        float alpha = 0.6f;
        float beta = 0.6f;
    };

    Scheduler();
    ~Scheduler();

    bool initialize();
    void cleanup();
    
    // Generate noise schedule
    std::vector<float> calculate_sigmas(const Config& config);
    
    // Scheduler type conversion
    static Type string_to_type(const std::string& name);
    static std::string type_to_string(Type type);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Individual scheduler implementations
    std::vector<float> normal_scheduler(const Config& config);
    std::vector<float> karras_scheduler(const Config& config);
    std::vector<float> exponential_scheduler(const Config& config);
    std::vector<float> sgm_uniform_scheduler(const Config& config);
    std::vector<float> simple_scheduler(const Config& config);
    std::vector<float> ddim_uniform_scheduler(const Config& config);
    std::vector<float> beta_scheduler(const Config& config);
    std::vector<float> linear_quadratic_scheduler(const Config& config);
    std::vector<float> kl_optimal_scheduler(const Config& config);
};
