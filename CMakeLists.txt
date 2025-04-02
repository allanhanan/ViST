cmake_minimum_required(VERSION 3.10)


project(ViT)

# Enable C++17 or later
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")

set(CMAKE_BUILD_TYPE Release)  # Use 'Release' build for optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -funroll-loops -fno-fat-lto-objects -ftree-vectorize")


# Include directories
include_directories(include)

# Source files
add_executable(ViT main.cpp)

