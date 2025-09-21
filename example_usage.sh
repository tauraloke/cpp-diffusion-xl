#!/bin/bash

echo "C++ Diffusion XL - Example Usage"
echo "================================="

# Check if executable exists
if [ ! -f "build/cpp_diffusion_xl" ]; then
    echo "Error: Executable not found. Please build the project first."
    echo "Run: ./build.sh"
    exit 1
fi

# Create results directory
mkdir -p results

echo ""
echo "Example 1: Basic generation"
echo "----------------------------"
./build/cpp_diffusion_xl --model "/path/to/sdxl/model" --prompt "a beautiful landscape with mountains and a lake" --width 1024 --height 1024 --steps 20 --cfg 7.0 --sampler euler_a --scheduler sgm_uniform --seed 42

echo ""
echo "Example 2: High quality generation"
echo "----------------------------------"
./build/cpp_diffusion_xl --model "/path/to/sdxl/model" --prompt "a detailed portrait of a woman, photorealistic, high quality" --negative "blurry, low quality, distorted" --width 1024 --height 1024 --steps 30 --cfg 8.0 --sampler euler_a --scheduler sgm_uniform --seed 123

echo ""
echo "Example 3: Fast generation"
echo "--------------------------"
./build/cpp_diffusion_xl --model "/path/to/sdxl/model" --prompt "a cute cat sitting on a windowsill" --width 512 --height 512 --steps 10 --cfg 5.0 --sampler euler_a --scheduler sgm_uniform --seed 456

echo ""
echo "All examples completed!"
echo "Check the 'results' folder for generated images."
