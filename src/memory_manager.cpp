#include "memory_manager.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <memory>
#include <vector>
#include <mutex>
#include <chrono>

class MemoryManager::Impl {
public:
    Impl() : cudnn_handle(nullptr), initialized(false) {}
    
    cudnnHandle_t cudnn_handle;
    bool initialized;
    VRAMState vram_state = VRAMState::NORMAL_VRAM;
    
    // Memory tracking
    size_t total_vram = 0;
    size_t total_ram = 0;
    size_t allocated_vram = 0;
    size_t allocated_ram = 0;
    
    // Thread safety
    std::mutex memory_mutex;
    
    // Memory allocation tracking
    struct Allocation {
        void* ptr;
        size_t size;
        bool is_vram;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::vector<Allocation> allocations;
    
    bool initialize_cuda() {
        int device_count;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
            std::cerr << "No CUDA devices found" << std::endl;
            return false;
        }
        
        if (cudaSetDevice(0) != cudaSuccess) {
            std::cerr << "Failed to set CUDA device" << std::endl;
            return false;
        }
        
        // Get device properties
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
            std::cerr << "Failed to get CUDA device properties" << std::endl;
            return false;
        }
        
        total_vram = prop.totalGlobalMem;
        std::cout << "CUDA Device: " << prop.name << std::endl;
        std::cout << "Total VRAM: " << (total_vram / (1024 * 1024)) << " MB" << std::endl;
        
        return true;
    }
    
    bool initialize_cudnn() {
        if (cudnnCreate(&cudnn_handle) != CUDNN_STATUS_SUCCESS) {
            std::cerr << "Failed to create cuDNN handle" << std::endl;
            return false;
        }
        
        // Enable cuDNN optimizations
        cudnnConvolutionDescriptor_t conv_desc;
        cudnnCreateConvolutionDescriptor(&conv_desc);
        // Use the new API for 2D convolution
        if (cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS) {
            std::cerr << "Warning: Failed to set convolution descriptor" << std::endl;
        }
        if (cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH) != CUDNN_STATUS_SUCCESS) {
            std::cerr << "Warning: Failed to set cuDNN math type" << std::endl;
        }
        cudnnDestroyConvolutionDescriptor(conv_desc);
        
        return true;
    }
    
    void cleanup_cudnn() {
        if (cudnn_handle) {
            cudnnDestroy(cudnn_handle);
            cudnn_handle = nullptr;
        }
    }
    
    void cleanup_allocations() {
        std::lock_guard<std::mutex> lock(memory_mutex);
        
        for (const auto& alloc : allocations) {
            if (alloc.is_vram) {
                cudaFree(alloc.ptr);
            } else {
                free(alloc.ptr);
            }
        }
        allocations.clear();
        allocated_vram = 0;
        allocated_ram = 0;
    }
    
    void track_allocation(void* ptr, size_t size, bool is_vram) {
        std::lock_guard<std::mutex> lock(memory_mutex);
        allocations.push_back({ptr, size, is_vram, std::chrono::steady_clock::now()});
        
        if (is_vram) {
            allocated_vram += size;
        } else {
            allocated_ram += size;
        }
    }
    
    void untrack_allocation(void* ptr) {
        std::lock_guard<std::mutex> lock(memory_mutex);
        
        auto it = std::find_if(allocations.begin(), allocations.end(),
            [ptr](const Allocation& alloc) { return alloc.ptr == ptr; });
        
        if (it != allocations.end()) {
            if (it->is_vram) {
                allocated_vram -= it->size;
            } else {
                allocated_ram -= it->size;
            }
            allocations.erase(it);
        }
    }
    
    size_t get_free_vram_impl() const {
        size_t free_mem, total_mem;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
            return free_mem;
        }
        return 0;
    }
    
    size_t get_free_ram_impl() const {
        // Simplified RAM detection
        return 8ULL * 1024 * 1024 * 1024; // Assume 8GB available
    }
    
    bool should_use_fp16_impl() const {
        // Check if device supports FP16
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            return prop.major >= 6; // Pascal and newer
        }
        return false;
    }
    
    bool should_use_bf16_impl() const {
        // Check if device supports BF16
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            return prop.major >= 8; // Ampere and newer
        }
        return false;
    }
    
    bool should_use_fp32_impl() const {
        // Always supported
        return true;
    }
    
    void* load_model_to_optimal_device_impl(size_t model_size) {
        size_t free_vram = get_free_vram_impl();
        size_t free_ram = get_free_ram_impl();
        
        // Determine optimal device based on available memory
        if (model_size < free_vram * 0.8f) {
            // Load to VRAM
            void* ptr = nullptr;
            if (cudaMalloc(&ptr, model_size) == cudaSuccess) {
                track_allocation(ptr, model_size, true);
                return ptr;
            }
        }
        
        // Fallback to RAM
        void* ptr = malloc(model_size);
        if (ptr) {
            track_allocation(ptr, model_size, false);
            return ptr;
        }
        
        return nullptr;
    }
    
    void unload_model_from_device_impl(void* model_ptr) {
        if (!model_ptr) return;
        
        // Find and remove allocation
        std::lock_guard<std::mutex> lock(memory_mutex);
        auto it = std::find_if(allocations.begin(), allocations.end(),
            [model_ptr](const Allocation& alloc) { return alloc.ptr == model_ptr; });
        
        if (it != allocations.end()) {
            if (it->is_vram) {
                cudaFree(model_ptr);
            } else {
                free(model_ptr);
            }
            allocations.erase(it);
        }
    }
};

MemoryManager& MemoryManager::getInstance() {
    static MemoryManager instance;
    return instance;
}

//MemoryManager::MemoryManager() : pImpl(std::make_unique<Impl>()) {}

//MemoryManager::~MemoryManager() {
//    cleanup();
//}

bool MemoryManager::initialize() {
    if (pImpl->initialized) {
        return true;
    }
    
    if (!pImpl->initialize_cuda()) {
        return false;
    }
    
    if (!pImpl->initialize_cudnn()) {
        return false;
    }
    
    pImpl->initialized = true;
    return true;
}

void MemoryManager::cleanup() {
    if (pImpl) {
        pImpl->cleanup_allocations();
        pImpl->cleanup_cudnn();
        pImpl->initialized = false;
    }
}

MemoryManager::VRAMState MemoryManager::get_vram_state() const {
    return pImpl->vram_state;
}

void MemoryManager::set_vram_state(VRAMState state) {
    pImpl->vram_state = state;
}

void* MemoryManager::allocate_vram(size_t size) {
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, size) == cudaSuccess) {
        pImpl->track_allocation(ptr, size, true);
        return ptr;
    }
    return nullptr;
}

void* MemoryManager::allocate_ram(size_t size) {
    void* ptr = malloc(size);
    if (ptr) {
        pImpl->track_allocation(ptr, size, false);
        return ptr;
    }
    return nullptr;
}

void MemoryManager::free_vram(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
        pImpl->untrack_allocation(ptr);
    }
}

void MemoryManager::free_ram(void* ptr) {
    if (ptr) {
        free(ptr);
        pImpl->untrack_allocation(ptr);
    }
}

size_t MemoryManager::get_total_vram() const {
    return pImpl->total_vram;
}

size_t MemoryManager::get_free_vram() const {
    return pImpl->get_free_vram_impl();
}

size_t MemoryManager::get_total_ram() const {
    return pImpl->total_ram;
}

size_t MemoryManager::get_free_ram() const {
    return pImpl->get_free_ram_impl();
}

bool MemoryManager::should_use_fp16() const {
    return pImpl->should_use_fp16_impl();
}

bool MemoryManager::should_use_bf16() const {
    return pImpl->should_use_bf16_impl();
}

bool MemoryManager::should_use_fp32() const {
    return pImpl->should_use_fp32_impl();
}

void* MemoryManager::load_model_to_optimal_device(size_t model_size) {
    return pImpl->load_model_to_optimal_device_impl(model_size);
}

void MemoryManager::unload_model_from_device(void* model_ptr) {
    pImpl->unload_model_from_device_impl(model_ptr);
}

void MemoryManager::soft_empty_cache() {
    cudaDeviceSynchronize();
    cudaDeviceReset();
}

void MemoryManager::cleanup_models() {
    pImpl->cleanup_allocations();
}