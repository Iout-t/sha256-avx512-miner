#ifndef MINER_H
#define MINER_H

#include <iostream>
#include <cstdint>
#include <immintrin.h>

// Align structure to a 64-byte boundary to perfectly match a CPU cache line
struct alignas(64) CryptoBlock {
    uint32_t words[16]; 
};

void precompute_midstate(const uint32_t* chunk1, uint32_t* output_midstate);

void execute_max_boundary_core(
    const uint32_t* midstate_raw,
    const CryptoBlock* multi_block_buffer,
    uint32_t target_difficulty
);

#endif // MINER_H
