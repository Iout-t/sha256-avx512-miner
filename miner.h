#ifndef MINER_H
#define MINER_H

#include <iostream>
#include <cstdint>
#include <immintrin.h>

// Align structures to 64-byte boundaries for optimal AVX-512 cache-line performance
struct alignas(64) BlockHeaderChunk2 {
    uint32_t data[16]; // Contains remaining Merkle root, timestamp, bits, and nonces
};

struct AVX512Midstate {
    __m512i h[8];      // 8 state variables (A-H), each storing 16 parallel channels
};

// Precalculates the first 64-byte chunk of the block header
void precompute_midstate(const uint32_t* chunk1, uint32_t* output_midstate);

// Vectorized mining loop processing 16 nonces simultaneously
void opencl_style_avx512_miner(
    const uint32_t* midstate_raw,
    const BlockHeaderChunk2* current_batch,
    const BlockHeaderChunk2* next_batch, // Used for asynchronous pre-fetching
    uint32_t base_nonce,
    uint32_t target_difficulty
);

#endif // MINER_H

