#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>
#include <memory>
#include <vector>

class MemoryManager {
public:
    enum class VRAMState {
        DISABLED = 0,    // No VRAM present
        NO_VRAM = 1,     // Very low VRAM
        LOW_VRAM = 2,
        NORMAL_VRAM = 3,
        HIGH_VRAM = 4,
        SHARED = 5       // Shared memory between CPU and GPU
    };

    static MemoryManager& getInstance();
    
    // Memory management
    bool initialize();
    void cleanup();
    
    // VRAM management
    VRAMState get_vram_state() const;
    void set_vram_state(VRAMState state);
    
    // Memory allocation
    void* allocate_vram(size_t size);
    void* allocate_ram(size_t size);
    void free_vram(void* ptr);
    void free_ram(void* ptr);
    
    // Memory info
    size_t get_total_vram() const;
    size_t get_free_vram() const;
    size_t get_total_ram() const;
    size_t get_free_ram() const;
    
    // Smart memory management
    bool should_use_fp16() const;
    bool should_use_bf16() const;
    bool should_use_fp32() const;
    
    // Model loading optimization
    void* load_model_to_optimal_device(size_t model_size);
    void unload_model_from_device(void* model_ptr);
    
    // Memory cleanup
    void soft_empty_cache();
    void cleanup_models();
    
private:
    MemoryManager() = default;
    ~MemoryManager() = default;
    
    class Impl;
    std::unique_ptr<Impl> pImpl;
};
