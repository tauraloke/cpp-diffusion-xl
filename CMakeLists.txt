cmake_minimum_required(VERSION 3.18)
project(cpp_diffusion_xl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add C++17 flag for MSVC
if(MSVC)
    add_compile_options(/std:c++17)
endif()

# Set OpenCV paths (update if your installation is different)
set(OpenCV_DIR "C:/opencv/build")

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV QUIET)

# Set CUDA and cuDNN paths
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0")
set(CUDNN_ROOT "C:/Program Files/NVIDIA/CUDNN/v9.13")
set(CUDNN_INCLUDE_DIR "${CUDNN_ROOT}/include/13.0")
set(CUDNN_LIBRARY_DIR "${CUDNN_ROOT}/lib/13.0/x64")
set(CUDNN_LIBRARY "${CUDNN_LIBRARY_DIR}/cudnn_cnn.lib")

if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn.h" AND EXISTS "${CUDNN_LIBRARY}")
    set(CUDNN_FOUND TRUE)
    message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
    message(STATUS "cuDNN include: ${CUDNN_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "cuDNN not found")
endif()

include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)
link_directories(${CUDAToolkit_LIBRARY_DIR})
link_directories(${CUDNN_LIBRARY_DIR})
link_directories(${OpenCV_DIR}/x64/vc17/lib)

# Source files
set(SOURCES
    src/main.cpp
    src/sdxl_model.cpp
    src/sampler.cpp
    src/scheduler.cpp
    src/memory_manager.cpp
    src/image_utils.cpp
    src/clip_encoder.cpp
    src/vae_decoder.cpp
    src/unet_model.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    CUDA::cudart
    CUDA::curand
    CUDA::cublas
    CUDA::cusparse
    ${CUDNN_LIBRARY}
    cudnn.lib
)


# Link OpenCV if found
if(OpenCV_FOUND)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV found, using OpenCV for image processing")
else()
    message(STATUS "OpenCV not found, using simple image processing")
endif()

# Compiler flags
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /O2 /W4 /std:c++17)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# CUDA specific flags
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75 80 86 89)

# Windows specific settings
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()