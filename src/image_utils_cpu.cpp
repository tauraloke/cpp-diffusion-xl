#include "image_utils.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>

bool ImageUtils::load_image(const std::string& path, ImageData& image) {
    // Simple image loading without OpenCV
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open image file: " << path << std::endl;
        return false;
    }
    
    // For simplicity, create a placeholder image
    image.width = 1024;
    image.height = 1024;
    image.channels = 3;
    image.is_gpu = false;
    
    // Allocate memory
    size_t size = image.width * image.height * image.channels * sizeof(float);
    image.data = new float[image.width * image.height * image.channels];
    
    // Fill with random data (placeholder)
    std::mt19937 gen(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < image.width * image.height * image.channels; i++) {
        image.data[i] = dist(gen);
    }
    
    return true;
}

bool ImageUtils::save_image(const std::string& path, const ImageData& image) {
    if (!image.data) {
        std::cerr << "No image data to save" << std::endl;
        return false;
    }
    
    // Simple PPM format saving
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to create image file: " << path << std::endl;
        return false;
    }
    
    // Write PPM header
    file << "P6\n" << image.width << " " << image.height << "\n255\n";
    
    // Write pixel data
    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            for (int c = 0; c < image.channels; c++) {
                int idx = (y * image.width + x) * image.channels + c;
                unsigned char pixel = static_cast<unsigned char>(std::clamp(image.data[idx] * 255.0f, 0.0f, 255.0f));
                file.write(reinterpret_cast<const char*>(&pixel), 1);
            }
        }
    }
    
    return true;
}

bool ImageUtils::save_image_png(const std::string& path, const ImageData& image) {
    // For now, save as PPM (PNG would require additional library)
    return save_image(path, image);
}

bool ImageUtils::normalize_image(ImageData& image, float min_val, float max_val) {
    if (!image.data) {
        return false;
    }
    
    size_t total_pixels = image.width * image.height * image.channels;
    
    // Find min and max values
    float current_min = image.data[0];
    float current_max = image.data[0];
    
    for (size_t i = 1; i < total_pixels; i++) {
        current_min = std::min(current_min, image.data[i]);
        current_max = std::max(current_max, image.data[i]);
    }
    
    // Normalize to [min_val, max_val]
    float range = current_max - current_min;
    if (range == 0.0f) {
        std::fill(image.data, image.data + total_pixels, min_val);
        return true;
    }
    
    float scale = (max_val - min_val) / range;
    float offset = min_val - current_min * scale;
    
    for (size_t i = 0; i < total_pixels; i++) {
        image.data[i] = image.data[i] * scale + offset;
    }
    
    return true;
}

bool ImageUtils::denormalize_image(ImageData& image, float min_val, float max_val) {
    return normalize_image(image, min_val, max_val);
}

bool ImageUtils::resize_image(const ImageData& input, ImageData& output, int new_width, int new_height) {
    if (!input.data) {
        return false;
    }
    
    // Allocate output
    if (!allocate_image(output, new_width, new_height, input.channels, input.is_gpu)) {
        return false;
    }
    
    // Simple nearest neighbor resize
    float scale_x = static_cast<float>(input.width) / new_width;
    float scale_y = static_cast<float>(input.height) / new_height;
    
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int src_x = static_cast<int>(x * scale_x);
            int src_y = static_cast<int>(y * scale_y);
            src_x = std::min(src_x, input.width - 1);
            src_y = std::min(src_y, input.height - 1);
            
            for (int c = 0; c < input.channels; c++) {
                int src_idx = (src_y * input.width + src_x) * input.channels + c;
                int dst_idx = (y * new_width + x) * input.channels + c;
                output.data[dst_idx] = input.data[src_idx];
            }
        }
    }
    
    return true;
}

bool ImageUtils::allocate_image(ImageData& image, int width, int height, int channels, bool use_gpu) {
    image.width = width;
    image.height = height;
    image.channels = channels;
    image.is_gpu = use_gpu;
    
    size_t size = width * height * channels * sizeof(float);
    
    if (use_gpu) {
        // No GPU support in CPU mode
        return false;
    } else {
        image.data = new float[width * height * channels];
        if (!image.data) {
            std::cerr << "Failed to allocate CPU memory for image" << std::endl;
            return false;
        }
    }
    
    return true;
}

void ImageUtils::free_image(ImageData& image) {
    if (image.data) {
        if (image.is_gpu) {
            // No GPU support in CPU mode
        } else {
            delete[] image.data;
        }
        image.data = nullptr;
    }
}

bool ImageUtils::gpu_to_cpu(const ImageData& gpu_image, ImageData& cpu_image) {
    // No GPU support in CPU mode
    return false;
}

bool ImageUtils::cpu_to_gpu(const ImageData& cpu_image, ImageData& gpu_image) {
    // No GPU support in CPU mode
    return false;
}

std::string ImageUtils::generate_timestamp_filename(const std::string& prefix, const std::string& extension) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << prefix << "_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << ms.count() << "." << extension;
    
    return ss.str();
}

bool ImageUtils::create_directory(const std::string& path) {
    try {
        return std::filesystem::create_directories(path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create directory: " << e.what() << std::endl;
        return false;
    }
}

bool ImageUtils::file_exists(const std::string& path) {
    try {
        return std::filesystem::exists(path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to check file existence: " << e.what() << std::endl;
        return false;
    }
}
