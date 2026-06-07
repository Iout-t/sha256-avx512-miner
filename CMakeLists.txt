cmake_minimum_required(VERSION 3.15)
project(SHA256_MAX_Boundary_Miner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -mavx512dq is strictly required for vpternlogd ternary logic
if(NOT MSVC)
    add_compile_options(
        -O3 
        -mavx512f 
        -mavx512vl 
        -mavx512dq 
        -march=native 
        -fomit-frame-pointer 
        -funroll-loops
        -flto
    )
else()
    add_compile_options(/O2 /arch:AVX512)
endif()

find_package(Threads REQUIRED)
include_directories(include)

add_executable(sha256_max_miner src/main.cpp src/miner.cpp)
target_link_libraries(sha256_max_miner PRIVATE Threads::Threads)
