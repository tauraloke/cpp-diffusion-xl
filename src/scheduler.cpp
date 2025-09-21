#include "scheduler.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

class Scheduler::Impl {
public:
    Impl() = default;
    
    // Normal scheduler implementation
    std::vector<float> normal_scheduler_impl(const Config& config) {
        std::vector<float> sigmas;
        sigmas.reserve(config.steps + 1);
        
        float start = config.sigma_max;
        float end = config.sigma_min;
        
        // Generate timesteps
        for (int i = 0; i < config.steps; i++) {
            float t = static_cast<float>(i) / (config.steps - 1);
            float sigma = start + t * (end - start);
            sigmas.push_back(sigma);
        }
        
        // Add zero at the end
        sigmas.push_back(0.0f);
        
        return sigmas;
    }
    
    // Karras scheduler implementation
    std::vector<float> karras_scheduler_impl(const Config& config) {
        std::vector<float> sigmas;
        sigmas.reserve(config.steps + 1);
        
        float min_inv_rho = std::pow(config.sigma_min, 1.0f / config.rho);
        float max_inv_rho = std::pow(config.sigma_max, 1.0f / config.rho);
        
        for (int i = 0; i < config.steps; i++) {
            float t = static_cast<float>(i) / (config.steps - 1);
            float inv_rho = max_inv_rho + t * (min_inv_rho - max_inv_rho);
            float sigma = std::pow(inv_rho, config.rho);
            sigmas.push_back(sigma);
        }
        
        sigmas.push_back(0.0f);
        return sigmas;
    }
    
    // Exponential scheduler implementation
    std::vector<float> exponential_scheduler_impl(const Config& config) {
        std::vector<float> sigmas;
        sigmas.reserve(config.steps + 1);
        
        float log_sigma_max = std::log(config.sigma_max);
        float log_sigma_min = std::log(config.sigma_min);
        
        for (int i = 0; i < config.steps; i++) {
            float t = static_cast<float>(i) / (config.steps - 1);
            float log_sigma = log_sigma_max + t * (log_sigma_min - log_sigma_max);
            float sigma = std::exp(log_sigma);
            sigmas.push_back(sigma);
        }
        
        sigmas.push_back(0.0f);
        return sigmas;
    }
    
    // SGM Uniform scheduler implementation (based on ComfyUI's sgm_uniform)
    std::vector<float> sgm_uniform_scheduler_impl(const Config& config) {
        std::vector<float> sigmas;
        sigmas.reserve(config.steps + 1);
        
        float start = config.sigma_max;
        float end = config.sigma_min;
        
        // Generate timesteps for SGM
        for (int i = 0; i < config.steps; i++) {
            float t = static_cast<float>(i) / config.steps; // Note: steps instead of steps-1 for SGM
            float sigma = start + t * (end - start);
            sigmas.push_back(sigma);
        }
        
        // Add zero at the end
        sigmas.push_back(0.0f);
        
        return sigmas;
    }
    
    // Simple scheduler implementation
    std::vector<float> simple_scheduler_impl(const Config& config) {
        std::vector<float> sigmas;
        sigmas.reserve(config.steps + 1);
        
        // Simple linear interpolation
        for (int i = 0; i < config.steps; i++) {
            float t = static_cast<float>(i) / (config.steps - 1);
            float sigma = config.sigma_max + t * (config.sigma_min - config.sigma_max);
            sigmas.push_back(sigma);
        }
        
        sigmas.push_back(0.0f);
        return sigmas;
    }
    
    // DDIM Uniform scheduler implementation
    std::vector<float> ddim_uniform_scheduler_impl(const Config& config) {
        std::vector<float> sigmas;
        sigmas.reserve(config.steps + 1);
        
        // DDIM uses a different approach
        float start = config.sigma_max;
        float end = config.sigma_min;
        
        // Check if we need to add an extra step
        bool append_zero = true;
        if (std::abs(end) < 1e-5f) {
            append_zero = false;
        }
        
        if (append_zero) {
            sigmas.push_back(0.0f);
        }
        
        // Generate steps
        int step_size = std::max(1, static_cast<int>(config.steps / config.steps));
        for (int i = 1; i < config.steps; i += step_size) {
            float t = static_cast<float>(i) / config.steps;
            float sigma = start + t * (end - start);
            sigmas.push_back(sigma);
        }
        
        // Reverse the order
        std::reverse(sigmas.begin(), sigmas.end());
        
        return sigmas;
    }
    
    // Beta scheduler implementation
    std::vector<float> beta_scheduler_impl(const Config& config) {
        std::vector<float> sigmas;
        sigmas.reserve(config.steps + 1);
        
        // Beta distribution parameters
        float alpha = config.alpha;
        float beta = config.beta;
        
        // Generate timesteps using beta distribution
        for (int i = 0; i < config.steps; i++) {
            float t = static_cast<float>(i) / (config.steps - 1);
            
            // Beta CDF inverse
            float beta_cdf_inv = 0.0f;
            if (t > 0.0f && t < 1.0f) {
                // Simplified beta CDF inverse
                beta_cdf_inv = std::pow(t, 1.0f / alpha) * std::pow(1.0f - t, 1.0f / beta);
            }
            
            float sigma = config.sigma_min + beta_cdf_inv * (config.sigma_max - config.sigma_min);
            sigmas.push_back(sigma);
        }
        
        sigmas.push_back(0.0f);
        return sigmas;
    }
    
    // Linear Quadratic scheduler implementation
    std::vector<float> linear_quadratic_scheduler_impl(const Config& config) {
        std::vector<float> sigmas;
        sigmas.reserve(config.steps + 1);
        
        if (config.steps == 1) {
            sigmas = {1.0f, 0.0f};
            return sigmas;
        }
        
        float threshold_noise = 0.025f;
        int linear_steps = config.steps / 2;
        
        // Linear part
        for (int i = 0; i < linear_steps; i++) {
            float sigma = i * threshold_noise / linear_steps;
            sigmas.push_back(sigma);
        }
        
        // Quadratic part
        float threshold_noise_step_diff = linear_steps - threshold_noise * config.steps;
        int quadratic_steps = config.steps - linear_steps;
        float quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps * quadratic_steps);
        float linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps * quadratic_steps);
        float const_term = quadratic_coef * (linear_steps * linear_steps);
        
        for (int i = linear_steps; i < config.steps; i++) {
            float sigma = quadratic_coef * (i * i) + linear_coef * i + const_term;
            sigmas.push_back(sigma);
        }
        
        // Transform to [0, 1] range
        for (auto& sigma : sigmas) {
            sigma = 1.0f - sigma;
        }
        
        // Scale by sigma_max
        for (auto& sigma : sigmas) {
            sigma *= config.sigma_max;
        }
        
        sigmas.push_back(0.0f);
        return sigmas;
    }
    
    // KL Optimal scheduler implementation
    std::vector<float> kl_optimal_scheduler_impl(const Config& config) {
        std::vector<float> sigmas;
        sigmas.reserve(config.steps + 1);
        
        // KL Optimal uses arctan transformation
        for (int i = 0; i < config.steps; i++) {
            float t = static_cast<float>(i) / (config.steps - 1);
            float sigma = std::tan(t * std::atan(config.sigma_min) + (1.0f - t) * std::atan(config.sigma_max));
            sigmas.push_back(sigma);
        }
        
        sigmas.push_back(0.0f);
        return sigmas;
    }
};

Scheduler::Scheduler() : pImpl(std::make_unique<Impl>()) {}

Scheduler::~Scheduler() = default;

bool Scheduler::initialize() {
    return true;
}

void Scheduler::cleanup() {
    // Nothing to cleanup
}

std::vector<float> Scheduler::calculate_sigmas(const Config& config) {
    switch (config.type) {
        case Type::NORMAL:
            return pImpl->normal_scheduler_impl(config);
        case Type::KARRAS:
            return pImpl->karras_scheduler_impl(config);
        case Type::EXPONENTIAL:
            return pImpl->exponential_scheduler_impl(config);
        case Type::SGM_UNIFORM:
            return pImpl->sgm_uniform_scheduler_impl(config);
        case Type::SIMPLE:
            return pImpl->simple_scheduler_impl(config);
        case Type::DDIM_UNIFORM:
            return pImpl->ddim_uniform_scheduler_impl(config);
        case Type::BETA:
            return pImpl->beta_scheduler_impl(config);
        case Type::LINEAR_QUADRATIC:
            return pImpl->linear_quadratic_scheduler_impl(config);
        case Type::KL_OPTIMAL:
            return pImpl->kl_optimal_scheduler_impl(config);
        default:
            std::cerr << "Unknown scheduler type, using SGM Uniform" << std::endl;
            return pImpl->sgm_uniform_scheduler_impl(config);
    }
}

Scheduler::Type Scheduler::string_to_type(const std::string& name) {
    if (name == "normal") return Type::NORMAL;
    if (name == "karras") return Type::KARRAS;
    if (name == "exponential") return Type::EXPONENTIAL;
    if (name == "sgm_uniform") return Type::SGM_UNIFORM;
    if (name == "simple") return Type::SIMPLE;
    if (name == "ddim_uniform") return Type::DDIM_UNIFORM;
    if (name == "beta") return Type::BETA;
    if (name == "linear_quadratic") return Type::LINEAR_QUADRATIC;
    if (name == "kl_optimal") return Type::KL_OPTIMAL;
    
    return Type::SGM_UNIFORM; // Default
}

std::string Scheduler::type_to_string(Type type) {
    switch (type) {
        case Type::NORMAL: return "normal";
        case Type::KARRAS: return "karras";
        case Type::EXPONENTIAL: return "exponential";
        case Type::SGM_UNIFORM: return "sgm_uniform";
        case Type::SIMPLE: return "simple";
        case Type::DDIM_UNIFORM: return "ddim_uniform";
        case Type::BETA: return "beta";
        case Type::LINEAR_QUADRATIC: return "linear_quadratic";
        case Type::KL_OPTIMAL: return "kl_optimal";
        default: return "sgm_uniform";
    }
}
