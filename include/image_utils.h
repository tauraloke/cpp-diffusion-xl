#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>

class ImageUtils {
public:
    struct ImageData {
        float* data = nullptr;
        int width = 0;
        int height = 0;
        int channels = 0;
        bool is_gpu = false;
        
        ~ImageData() {
            if (data) {
                if (is_gpu) {
                    cudaFree(data);
                } else {
                    delete[] data;
                }
            }
        }
    };

    // Image loading/saving
    static bool load_image(const std::string& path, ImageData& image);
    static bool save_image(const std::string& path, const ImageData& image);
    static bool save_image_png(const std::string& path, const ImageData& image);
    
    // Image processing
    static bool normalize_image(ImageData& image, float min_val = 0.0f, float max_val = 1.0f);
    static bool denormalize_image(ImageData& image, float min_val = 0.0f, float max_val = 1.0f);
    static bool resize_image(const ImageData& input, ImageData& output, int new_width, int new_height);
    
    // Memory management
    static bool allocate_image(ImageData& image, int width, int height, int channels, bool use_gpu = false);
    static void free_image(ImageData& image);
    
    // Data conversion
    static bool gpu_to_cpu(const ImageData& gpu_image, ImageData& cpu_image);
    static bool cpu_to_gpu(const ImageData& cpu_image, ImageData& gpu_image);
    
    // Utility functions
    static std::string generate_timestamp_filename(const std::string& prefix = "image", const std::string& extension = "png");
    static bool create_directory(const std::string& path);
    static bool file_exists(const std::string& path);
};
