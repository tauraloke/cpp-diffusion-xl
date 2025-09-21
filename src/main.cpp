#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>

#include "sdxl_model.h"
#include "image_utils.h"

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --model <path>           Path to SDXL model file (required)\n";
    std::cout << "  --prompt <text>          Positive prompt (required)\n";
    std::cout << "  --negative <text>        Negative prompt (optional)\n";
    std::cout << "  --width <int>            Image width (default: 1024)\n";
    std::cout << "  --height <int>           Image height (default: 1024)\n";
    std::cout << "  --steps <int>            Number of sampling steps (default: 20)\n";
    std::cout << "  --cfg <float>            CFG scale (default: 7.0)\n";
    std::cout << "  --sampler <name>         Sampler type (default: euler_a)\n";
    std::cout << "  --scheduler <name>       Scheduler type (default: sgm_uniform)\n";
    std::cout << "  --seed <int>             Random seed (default: 0)\n";
    std::cout << "  --output <path>          Output directory (default: ./results)\n";
    std::cout << "  --help                   Show this help message\n";
    std::cout << "\n";
    std::cout << "Available samplers: euler_a, euler, heun, dpm_2, dpm_2_ancestral, lms, dpm_fast, dpm_adaptive, dpmpp_2s_ancestral, dpmpp_sde, dpmpp_2m, dpmpp_2m_sde, ddpm, lcm\n";
    std::cout << "Available schedulers: normal, karras, exponential, sgm_uniform, simple, ddim_uniform, beta, linear_quadratic, kl_optimal\n";
}

std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

int main(int argc, char* argv[]) {
    // Default configuration
    SDXLModel::Config config;
    std::string output_dir = "./results";
    bool help_requested = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            help_requested = true;
        } else if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            config.positive_prompt = argv[++i];
        } else if (arg == "--negative" && i + 1 < argc) {
            config.negative_prompt = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            config.width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            config.height = std::stoi(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            config.steps = std::stoi(argv[++i]);
        } else if (arg == "--cfg" && i + 1 < argc) {
            config.cfg_scale = std::stof(argv[++i]);
        } else if (arg == "--sampler" && i + 1 < argc) {
            config.sampler = argv[++i];
        } else if (arg == "--scheduler" && i + 1 < argc) {
            config.scheduler = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = std::stoul(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (help_requested) {
        print_usage(argv[0]);
        return 0;
    }
    
    // Validate required arguments
    if (config.model_path.empty()) {
        std::cerr << "Error: --model is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    if (config.positive_prompt.empty()) {
        std::cerr << "Error: --prompt is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Create output directory
    if (!ImageUtils::create_directory(output_dir)) {
        std::cerr << "Error: Failed to create output directory: " << output_dir << std::endl;
        return 1;
    }
    
    std::cout << "=== C++ Diffusion XL ===" << std::endl;
    std::cout << "Model: " << config.model_path << std::endl;
    std::cout << "Prompt: " << config.positive_prompt << std::endl;
    std::cout << "Negative: " << config.negative_prompt << std::endl;
    std::cout << "Size: " << config.width << "x" << config.height << std::endl;
    std::cout << "Steps: " << config.steps << std::endl;
    std::cout << "CFG: " << config.cfg_scale << std::endl;
    std::cout << "Sampler: " << config.sampler << std::endl;
    std::cout << "Scheduler: " << config.scheduler << std::endl;
    std::cout << "Seed: " << config.seed << std::endl;
    std::cout << "Output: " << output_dir << std::endl;
    std::cout << std::endl;
    
    try {
        // Initialize SDXL model
        SDXLModel model;
        
        std::cout << "Loading model..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (!model.load_model(config.model_path)) {
            std::cerr << "Error: Failed to load model" << std::endl;
            return 1;
        }
        
        auto load_time = std::chrono::high_resolution_clock::now();
        auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_time - start_time);
        std::cout << "Model loaded in " << load_duration.count() << "ms" << std::endl;
        
        // Generate image
        std::cout << "Generating image..." << std::endl;
        auto gen_start = std::chrono::high_resolution_clock::now();
        
        std::string timestamp = get_current_timestamp();
        std::string output_path = output_dir + "/" + timestamp + ".png";
        
        if (!model.generate_image(config, output_path)) {
            std::cerr << "Error: Failed to generate image" << std::endl;
            return 1;
        }
        
        auto gen_end = std::chrono::high_resolution_clock::now();
        auto gen_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start);
        
        std::cout << "Image generated in " << gen_duration.count() << "ms" << std::endl;
        std::cout << "Saved to: " << output_path << std::endl;
        
        auto total_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_time - start_time);
        std::cout << "Total time: " << total_duration.count() << "ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
