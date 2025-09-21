cmake_minimum_required(VERSION 3.18)
project(cpp_diffusion_xl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenCV QUIET)
find_package(PkgConfig REQUIRED)

# Set CUDA and cuDNN paths
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0")
set(CUDNN_ROOT "C:/Program Files/NVIDIA/CUDNN/v9.13")

# Find cuDNN
find_path(CUDNN_INCLUDE_DIR cudnn.h
    PATHS
    ${CUDNN_ROOT}/include/13.0
    ${CUDNN_ROOT}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    NO_DEFAULT_PATH
)

find_library(CUDNN_LIBRARY cudnn
    PATHS
    ${CUDNN_ROOT}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
    NO_DEFAULT_PATH
)

if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    set(CUDNN_FOUND TRUE)
    message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
    message(STATUS "cuDNN include: ${CUDNN_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "cuDNN not found")
endif()

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

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
    ${CUDA_LIBRARIES}
    ${CUDNN_LIBRARY}
    cudart
    curand
    cublas
    cusparse
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
target_compile_options(${PROJECT_NAME} PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
)

# CUDA specific flags
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75 80 86 89)

# Windows specific settings
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
    set_property(TARGET ${PROJECT_NAME} PROPERTY WIN32_EXECUTABLE TRUE)
endif()