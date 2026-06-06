cmake_minimum_required(VERSION 3.15)
project(SHA256_AVX512_Miner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force high-level optimization and enable AVX-512 instruction sets
if(MSVC)
    add_compile_options(/O2 /arch:AVX512)
else()
    add_compile_options(-O3 -mavx512f -mavx512vl -march=native -fomit-frame-pointer)
endif()

include_directories(include)

add_executable(sha256_miner src/main.cpp src/miner.cpp)

