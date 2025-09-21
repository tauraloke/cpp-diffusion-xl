#include "sdxl_model.h"
#include "memory_manager.h"
#include "clip_encoder.h"
#include "vae_decoder.h"
#include "unet_model.h"
#include "sampler.h"
#include "scheduler.h"
#include "image_utils.h"
#include <iostream>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

class SDXLModel::Impl {
public:
    Impl() : memory_manager(MemoryManager::getInstance()) {}
    
    MemoryManager& memory_manager;
    std::unique_ptr<CLIPEncoder> clip_encoder;
    std::unique_ptr<VAEDecoder> vae_decoder;
    std::unique_ptr<UNetModel> unet_model;
    std::unique_ptr<Sampler> sampler;
    std::unique_ptr<Scheduler> scheduler;
    
    bool model_loaded = false;
    Config current_config;
    
    // Memory management
    void* unet_weights = nullptr;
    void* clip_weights = nullptr;
    void* vae_weights = nullptr;
    
    // Generation state
    float* noise = nullptr;
    float* latents = nullptr;
    float* text_embeddings = nullptr;
    float* timesteps = nullptr;
    std::vector<float> sigmas;
    
    bool allocate_generation_memory(const Config& config) {
        // Calculate memory requirements
        int latent_height = config.height / 8;
        int latent_width = config.width / 8;
        int latent_channels = 4;
        int batch_size = 1;
        
        size_t noise_size = batch_size * latent_channels * latent_height * latent_width * sizeof(float);
        size_t latents_size = noise_size;
        size_t text_embeddings_size = batch_size * 77 * 2048 * sizeof(float); // SDXL uses 2048 for text embedding
        size_t timesteps_size = config.steps * sizeof(float);
        
        // Allocate GPU memory
        if (cudaMalloc(&noise, noise_size) != cudaSuccess) {
            std::cerr << "Failed to allocate noise memory" << std::endl;
            return false;
        }
        
        if (cudaMalloc(&latents, latents_size) != cudaSuccess) {
            std::cerr << "Failed to allocate latents memory" << std::endl;
            return false;
        }
        
        if (cudaMalloc(&text_embeddings, text_embeddings_size) != cudaSuccess) {
            std::cerr << "Failed to allocate text embeddings memory" << std::endl;
            return false;
        }
        
        if (cudaMalloc(&timesteps, timesteps_size) != cudaSuccess) {
            std::cerr << "Failed to allocate timesteps memory" << std::endl;
            return false;
        }
        
        return true;
    }
    
    void free_generation_memory() {
        if (noise) { cudaFree(noise); noise = nullptr; }
        if (latents) { cudaFree(latents); latents = nullptr; }
        if (text_embeddings) { cudaFree(text_embeddings); text_embeddings = nullptr; }
        if (timesteps) { cudaFree(timesteps); timesteps = nullptr; }
    }
    
    bool generate_noise(const Config& config) {
        int latent_height = config.height / 8;
        int latent_width = config.width / 8;
        int latent_channels = 4;
        int batch_size = 1;
        
        // Generate random noise
        std::mt19937 gen(config.seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        std::vector<float> noise_cpu(batch_size * latent_channels * latent_height * latent_width);
        for (auto& val : noise_cpu) {
            val = dist(gen);
        }
        
        // Copy to GPU
        if (cudaMemcpy(noise, noise_cpu.data(), noise_cpu.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "Failed to copy noise to GPU" << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool encode_text(const Config& config) {
        if (!clip_encoder) {
            std::cerr << "CLIP encoder not loaded" << std::endl;
            return false;
        }
        
        // Encode positive prompt
        std::vector<float> pos_embeddings(77 * 2048);
        if (!clip_encoder->encode_text(config.positive_prompt, pos_embeddings.data())) {
            std::cerr << "Failed to encode positive prompt" << std::endl;
            return false;
        }
        
        // Encode negative prompt
        std::vector<float> neg_embeddings(77 * 2048);
        if (!clip_encoder->encode_text(config.negative_prompt, neg_embeddings.data())) {
            std::cerr << "Failed to encode negative prompt" << std::endl;
            return false;
        }
        
        // Concatenate embeddings (positive + negative)
        std::vector<float> combined_embeddings(2 * 77 * 2048);
        std::copy(pos_embeddings.begin(), pos_embeddings.end(), combined_embeddings.begin());
        std::copy(neg_embeddings.begin(), neg_embeddings.end(), combined_embeddings.begin() + 77 * 2048);
        
        // Copy to GPU
        if (cudaMemcpy(text_embeddings, combined_embeddings.data(), combined_embeddings.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "Failed to copy text embeddings to GPU" << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool perform_sampling(const Config& config) {
        if (!unet_model || !sampler || !scheduler) {
            std::cerr << "Required models not loaded" << std::endl;
            return false;
        }
        
        // Generate noise schedule
        Scheduler::Config sched_config;
        sched_config.type = Scheduler::string_to_type(config.scheduler);
        sched_config.steps = config.steps;
        sched_config.sigma_min = 0.0292f;
        sched_config.sigma_max = 14.6146f;
        
        sigmas = scheduler->calculate_sigmas(sched_config);
        if (sigmas.empty()) {
            std::cerr << "Failed to calculate sigmas" << std::endl;
            return false;
        }
        
        // Copy sigmas to GPU
        if (cudaMemcpy(timesteps, sigmas.data(), sigmas.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "Failed to copy sigmas to GPU" << std::endl;
            return false;
        }
        
        // Configure sampler
        Sampler::Config sampler_config;
        sampler_config.type = Sampler::string_to_type(config.sampler);
        sampler_config.steps = config.steps;
        sampler_config.seed = config.seed;
        
        // Perform sampling
        int latent_height = config.height / 8;
        int latent_width = config.width / 8;
        int latent_channels = 4;
        int batch_size = 1;
        
        // Copy noise to latents
        if (cudaMemcpy(latents, noise, batch_size * latent_channels * latent_height * latent_width * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
            std::cerr << "Failed to copy noise to latents" << std::endl;
            return false;
        }
        
        // Run sampling loop
        for (int i = 0; i < config.steps; i++) {
            float current_sigma = sigmas[i];
            float next_sigma = (i < config.steps - 1) ? sigmas[i + 1] : 0.0f;
            
            // UNet forward pass
            if (!unet_model->forward(latents, &current_sigma, text_embeddings, nullptr, latents, batch_size)) {
                std::cerr << "UNet forward pass failed at step " << i << std::endl;
                return false;
            }
            
            // Apply CFG if needed
            if (config.cfg_scale > 1.0f) {
                // This is a simplified CFG implementation
                // In practice, you'd need to run both positive and negative predictions
                // and blend them according to the CFG scale
            }
            
            // Sampler step
            if (!sampler->sample(unet_model.get(), latents, timesteps, batch_size, latent_channels, latent_height, latent_width, sampler_config)) {
                std::cerr << "Sampling step failed at step " << i << std::endl;
                return false;
            }
            
            // Progress callback
            if (i % 5 == 0 || i == config.steps - 1) {
                std::cout << "Step " << (i + 1) << "/" << config.steps << " (sigma: " << current_sigma << ")" << std::endl;
            }
        }
        
        return true;
    }
    
    bool decode_latents(const Config& config, const std::string& output_path) {
        if (!vae_decoder) {
            std::cerr << "VAE decoder not loaded" << std::endl;
            return false;
        }
        
        // Decode latents to image
        int latent_height = config.height / 8;
        int latent_width = config.width / 8;
        int latent_channels = 4;
        int batch_size = 1;
        
        // Allocate output image memory
        ImageUtils::ImageData output_image;
        if (!ImageUtils::allocate_image(output_image, config.width, config.height, 3, true)) {
            std::cerr << "Failed to allocate output image memory" << std::endl;
            return false;
        }
        
        // Decode
        if (!vae_decoder->decode(latents, output_image.data, batch_size)) {
            std::cerr << "Failed to decode latents" << std::endl;
            return false;
        }
        
        // Convert to CPU for saving
        ImageUtils::ImageData cpu_image;
        if (!ImageUtils::gpu_to_cpu(output_image, cpu_image)) {
            std::cerr << "Failed to convert image to CPU" << std::endl;
            return false;
        }
        
        // Normalize image to 0-255 range
        if (!ImageUtils::denormalize_image(cpu_image, 0.0f, 255.0f)) {
            std::cerr << "Failed to normalize image" << std::endl;
            return false;
        }
        
        // Save image
        if (!ImageUtils::save_image_png(output_path, cpu_image)) {
            std::cerr << "Failed to save image" << std::endl;
            return false;
        }
        
        return true;
    }
};

SDXLModel::SDXLModel() : pImpl(std::make_unique<Impl>()) {}

SDXLModel::~SDXLModel() {
    if (pImpl) {
        pImpl->free_generation_memory();
        if (pImpl->unet_weights) pImpl->memory_manager.free_vram(pImpl->unet_weights);
        if (pImpl->clip_weights) pImpl->memory_manager.free_vram(pImpl->clip_weights);
        if (pImpl->vae_weights) pImpl->memory_manager.free_vram(pImpl->vae_weights);
    }
}

bool SDXLModel::load_model(const std::string& model_path) {
    if (!pImpl->memory_manager.initialize()) {
        std::cerr << "Failed to initialize memory manager" << std::endl;
        return false;
    }
    
    // Initialize components
    pImpl->clip_encoder = std::make_unique<CLIPEncoder>();
    pImpl->vae_decoder = std::make_unique<VAEDecoder>();
    pImpl->unet_model = std::make_unique<UNetModel>();
    pImpl->sampler = std::make_unique<Sampler>();
    pImpl->scheduler = std::make_unique<Scheduler>();
    
    // Load models
    if (!pImpl->clip_encoder->load_model(model_path + "/text_encoder")) {
        std::cerr << "Failed to load CLIP encoder" << std::endl;
        return false;
    }
    
    if (!pImpl->vae_decoder->load_model(model_path + "/vae")) {
        std::cerr << "Failed to load VAE decoder" << std::endl;
        return false;
    }
    
    if (!pImpl->unet_model->load_model(model_path + "/unet")) {
        std::cerr << "Failed to load UNet model" << std::endl;
        return false;
    }
    
    if (!pImpl->sampler->initialize()) {
        std::cerr << "Failed to initialize sampler" << std::endl;
        return false;
    }
    
    if (!pImpl->scheduler->initialize()) {
        std::cerr << "Failed to initialize scheduler" << std::endl;
        return false;
    }
    
    pImpl->model_loaded = true;
    return true;
}

bool SDXLModel::generate_image(const Config& config, const std::string& output_path) {
    if (!pImpl->model_loaded) {
        std::cerr << "Model not loaded" << std::endl;
        return false;
    }
    
    pImpl->current_config = config;
    
    // Allocate generation memory
    if (!pImpl->allocate_generation_memory(config)) {
        std::cerr << "Failed to allocate generation memory" << std::endl;
        return false;
    }
    
    try {
        // Generate noise
        if (!pImpl->generate_noise(config)) {
            std::cerr << "Failed to generate noise" << std::endl;
            return false;
        }
        
        // Encode text
        if (!pImpl->encode_text(config)) {
            std::cerr << "Failed to encode text" << std::endl;
            return false;
        }
        
        // Perform sampling
        if (!pImpl->perform_sampling(config)) {
            std::cerr << "Failed to perform sampling" << std::endl;
            return false;
        }
        
        // Decode latents to image
        if (!pImpl->decode_latents(config, output_path)) {
            std::cerr << "Failed to decode latents" << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error during generation: " << e.what() << std::endl;
        return false;
    }
    
    return true;
}

void SDXLModel::set_memory_fraction(float fraction) {
    // Implementation would set CUDA memory fraction
}

void SDXLModel::enable_mixed_precision(bool enable) {
    // Implementation would enable/disable mixed precision
}
