#include "memory_manager.h"
#include <iostream>
#include <algorithm>
#include <thread>
#include <mutex>

class MemoryManager::Impl {
public:
    Impl() : initialized(false) {}
    
    bool initialized;
    VRAMState vram_state = VRAMState::DISABLED; // CPU only mode
    
    // Memory tracking
    size_t total_ram = 0;
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
    
    bool initialize_cpu() {
        // Get system RAM info
        total_ram = 8ULL * 1024 * 1024 * 1024; // Assume 8GB
        std::cout << "CPU-only mode: " << (total_ram / (1024 * 1024)) << " MB RAM available" << std::endl;
        return true;
    }
    
    void cleanup_allocations() {
        std::lock_guard<std::mutex> lock(memory_mutex);
        
        for (const auto& alloc : allocations) {
            if (alloc.is_vram) {
                // No VRAM in CPU mode
            } else {
                free(alloc.ptr);
            }
        }
        allocations.clear();
        allocated_ram = 0;
    }
    
    void track_allocation(void* ptr, size_t size, bool is_vram) {
        std::lock_guard<std::mutex> lock(memory_mutex);
        allocations.push_back({ptr, size, is_vram, std::chrono::steady_clock::now()});
        
        if (!is_vram) {
            allocated_ram += size;
        }
    }
    
    void untrack_allocation(void* ptr) {
        std::lock_guard<std::mutex> lock(memory_mutex);
        
        auto it = std::find_if(allocations.begin(), allocations.end(),
            [ptr](const Allocation& alloc) { return alloc.ptr == ptr; });
        
        if (it != allocations.end()) {
            if (!it->is_vram) {
                allocated_ram -= it->size;
            }
            allocations.erase(it);
        }
    }
    
    size_t get_free_ram_impl() const {
        // Simplified RAM detection
        return total_ram - allocated_ram;
    }
    
    bool should_use_fp16_impl() const {
        return false; // CPU doesn't support FP16 efficiently
    }
    
    bool should_use_bf16_impl() const {
        return false; // CPU doesn't support BF16 efficiently
    }
    
    bool should_use_fp32_impl() const {
        return true; // Always use FP32 on CPU
    }
    
    void* load_model_to_optimal_device_impl(size_t model_size) {
        // Always load to RAM in CPU mode
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
            if (!it->is_vram) {
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

MemoryManager::MemoryManager() : pImpl(std::make_unique<Impl>()) {}

MemoryManager::~MemoryManager() {
    cleanup();
}

bool MemoryManager::initialize() {
    if (pImpl->initialized) {
        return true;
    }
    
    if (!pImpl->initialize_cpu()) {
        return false;
    }
    
    pImpl->initialized = true;
    return true;
}

void MemoryManager::cleanup() {
    if (pImpl) {
        pImpl->cleanup_allocations();
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
    // No VRAM in CPU mode
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
    // No VRAM in CPU mode
}

void MemoryManager::free_ram(void* ptr) {
    if (ptr) {
        free(ptr);
        pImpl->untrack_allocation(ptr);
    }
}

size_t MemoryManager::get_total_vram() const {
    return 0; // No VRAM in CPU mode
}

size_t MemoryManager::get_free_vram() const {
    return 0; // No VRAM in CPU mode
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
    // No cache to empty in CPU mode
}

void MemoryManager::cleanup_models() {
    pImpl->cleanup_allocations();
}
