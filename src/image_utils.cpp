#include "image_utils.h"
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>

bool ImageUtils::load_image(const std::string& path, ImageData& image) {
    cv::Mat cv_image = cv::imread(path, cv::IMREAD_COLOR);
    if (cv_image.empty()) {
        std::cerr << "Failed to load image: " << path << std::endl;
        return false;
    }
    
    // Convert BGR to RGB
    cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2RGB);
    
    image.width = cv_image.cols;
    image.height = cv_image.rows;
    image.channels = cv_image.channels();
    image.is_gpu = false;
    
    // Allocate memory
    size_t size = image.width * image.height * image.channels * sizeof(float);
    image.data = new float[image.width * image.height * image.channels];
    
    // Convert to float and normalize to [0, 1]
    cv_image.convertTo(cv_image, CV_32F, 1.0 / 255.0);
    
    // Copy data
    std::memcpy(image.data, cv_image.data, size);
    
    return true;
}

bool ImageUtils::save_image(const std::string& path, const ImageData& image) {
    if (!image.data) {
        std::cerr << "No image data to save" << std::endl;
        return false;
    }
    
    // Convert to OpenCV Mat
    cv::Mat cv_image;
    if (image.is_gpu) {
        // Copy from GPU to CPU first
        ImageData cpu_image;
        if (!gpu_to_cpu(image, cpu_image)) {
            return false;
        }
        
        cv_image = cv::Mat(image.height, image.width, CV_32FC3, cpu_image.data);
    } else {
        cv_image = cv::Mat(image.height, image.width, CV_32FC3, image.data);
    }
    
    // Convert RGB to BGR
    cv::cvtColor(cv_image, cv_image, cv::COLOR_RGB2BGR);
    
    // Convert to uint8
    cv::Mat cv_image_uint8;
    cv_image.convertTo(cv_image_uint8, CV_8U, 255.0);
    
    // Save image
    return cv::imwrite(path, cv_image_uint8);
}

bool ImageUtils::save_image_png(const std::string& path, const ImageData& image) {
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
    
    if (input.is_gpu) {
        // GPU resize would use CUDA kernels
        // For now, copy to CPU, resize, then copy back
        ImageData cpu_input, cpu_output;
        if (!gpu_to_cpu(input, cpu_input)) {
            return false;
        }
        
        if (!resize_image(cpu_input, cpu_output, new_width, new_height)) {
            return false;
        }
        
        return cpu_to_gpu(cpu_output, output);
    } else {
        // CPU resize using OpenCV
        cv::Mat cv_input(input.height, input.width, CV_32FC3, input.data);
        cv::Mat cv_output;
        cv::resize(cv_input, cv_output, cv::Size(new_width, new_height));
        
        std::memcpy(output.data, cv_output.data, new_width * new_height * input.channels * sizeof(float));
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
        if (cudaMalloc(&image.data, size) != cudaSuccess) {
            std::cerr << "Failed to allocate GPU memory for image" << std::endl;
            return false;
        }
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
            cudaFree(image.data);
        } else {
            delete[] image.data;
        }
        image.data = nullptr;
    }
}

bool ImageUtils::gpu_to_cpu(const ImageData& gpu_image, ImageData& cpu_image) {
    if (!gpu_image.data || !gpu_image.is_gpu) {
        return false;
    }
    
    if (!allocate_image(cpu_image, gpu_image.width, gpu_image.height, gpu_image.channels, false)) {
        return false;
    }
    
    size_t size = gpu_image.width * gpu_image.height * gpu_image.channels * sizeof(float);
    if (cudaMemcpy(cpu_image.data, gpu_image.data, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Failed to copy image from GPU to CPU" << std::endl;
        free_image(cpu_image);
        return false;
    }
    
    return true;
}

bool ImageUtils::cpu_to_gpu(const ImageData& cpu_image, ImageData& gpu_image) {
    if (!cpu_image.data || cpu_image.is_gpu) {
        return false;
    }
    
    if (!allocate_image(gpu_image, cpu_image.width, cpu_image.height, cpu_image.channels, true)) {
        return false;
    }
    
    size_t size = cpu_image.width * cpu_image.height * cpu_image.channels * sizeof(float);
    if (cudaMemcpy(gpu_image.data, cpu_image.data, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Failed to copy image from CPU to GPU" << std::endl;
        free_image(gpu_image);
        return false;
    }
    
    return true;
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