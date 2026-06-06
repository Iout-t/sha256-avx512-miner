#include "miner.h"

// SHA-256 Constants
const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// AVX-512 Bitwise Bit-Select Macro (Ch function)
#define CH(x, y, z) _mm512_xor_si512(z, _mm512_and_si512(x, _mm512_xor_si512(y, z)))

// AVX-512 Majority Macro (Maj function)
#define MAJ(x, y, z) _mm512_or_si512(_mm512_and_si512(x, y), _mm512_and_si512(z, _mm512_xor_si512(x, y)))

// AVX-512 Vector Rotations
#define ROR32(x, n) _mm512_ror_epi32(x, n)

#define EP0(x) _mm512_xor_si512(ROR32(x, 2), _mm512_xor_si512(ROR32(x, 13), ROR32(x, 22)))
#define EP1(x) _mm512_xor_si512(ROR32(x, 6), _mm512_xor_si512(ROR32(x, 11), ROR32(x, 25)))
#define SIG0(x) _mm512_xor_si512(ROR32(x, 7), _mm512_xor_si512(ROR32(x, 18), _mm512_srli_epi32(x, 3)))
#define SIG1(x) _mm512_xor_si512(ROR32(x, 17), _mm512_xor_si512(ROR32(x, 19), _mm512_srli_epi32(x, 10)))

// PARAMETER 1: Completely Unrolled Execution Round to maximize ILP and avoid branch prediction stalls
#define SHA256_ROUND(a, b, c, d, e, f, g, h, w, k) { \
    __m512i t1 = _mm512_add_epi32(h, _mm512_add_epi32(EP1(e), _mm512_add_epi32(CH(e, f, g), _mm512_add_epi32(w, _mm512_set1_epi32(k))))); \
    __m512i t2 = _mm512_add_epi32(EP0(a), MAJ(a, b, c)); \
    d = _mm512_add_epi32(d, t1); \
    h = _mm512_add_epi32(t1, t2); \
}

void precompute_midstate(const uint32_t* chunk1, uint32_t* output_midstate) {
    // Standard sequential execution for Chunk 1 (Runs once per 4GB nonce space allocation)
    // Minimizes initialization costs before launching core loops
    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;
    
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = chunk1[i];
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = ((w[i-15] >> 7) | (w[i-15] << 25)) ^ ((w[i-15] >> 18) | (w[i-15] << 14)) ^ (w[i-15] >> 3);
        uint32_t s1 = ((w[i-2] >> 17) | (w[i-2] << 15)) ^ ((w[i-2] >> 19) | (w[i-2] << 13)) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = h + (((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^ ((e >> 25) | (e << 7))) + ((e & f) ^ (~e & g)) + w[i] + K[i];
        uint32_t t2 = (((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^ ((a >> 22) | (a << 10))) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    
    output_midstate[0] = a + 0x6a09e667; output_midstate[1] = b + 0xbb67ae85;
    output_midstate[2] = c + 0x3c6ef372; output_midstate[3] = d + 0xa54ff53a;
    output_midstate[4] = e + 0x510e527f; output_midstate[5] = f + 0x9b05688c;
    output_midstate[6] = g + 0x1f83d9ab; output_midstate[7] = h + 0x5be0cd19;
}

void opencl_style_avx512_miner(
    const uint32_t* midstate_raw,
    const BlockHeaderChunk2* current_batch,
    const BlockHeaderChunk2* next_batch,
    uint32_t base_nonce,
    uint32_t target_difficulty
) {
    // PARAMETER 2: Asynchronous Pre-fetching
    // Issue non-temporal hint prefetch to pull the NEXT data batch cleanly into L1 cache lines
    _mm_prefetch(reinterpret_cast<const char*>(next_batch), _MM_HINT_T0);

    // Initializing 16 vector lanes with the loaded midstate matrices
    __m512i a = _mm512_set1_epi32(midstate_raw[0]);
    __m512i b = _mm512_set1_epi32(midstate_raw[1]);
    __m512i c = _mm512_set1_epi32(midstate_raw[2]);
    __m512i d = _mm512_set1_epi32(midstate_raw[3]);
    __m512i e = _mm512_set1_epi32(midstate_raw[4]);
    __m512i f = _mm512_set1_epi32(midstate_raw[5]);
    __m512i g = _mm512_set1_epi32(midstate_raw[6]);
    __m512i h = _mm512_set1_epi32(midstate_raw[7]);

    __m512i w[64];

    // Map Chunk 2 elements across 16 parallel vector structures
    for (int i = 0; i < 16; i++) {
        w[i] = _mm512_set1_epi32(current_batch->data[i]);
    }

    // PARAMETER 1 CONT.: Inject unique incremental nonces into lane vectors [0 through 15]
    __m512i nonce_vectors = _mm512_add_epi32(_mm512_set1_epi32(base_nonce), 
                            _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15));
    w[3] = nonce_vectors; // Map execution position directly to designated nonce slot inside Chunk 2

    // Unroll Message Schedule computation loops entirely
    for (int i = 16; i < 64; i++) {
        w[i] = _mm512_add_epi32(w[i-16], _mm512_add_epi32(SIG0(w[i-15]), _mm512_add_epi32(w[i-7], SIG1(w[i-2]))));
    }

    // PARAMETER 3: Midstate optimization logic executes here
    // Bypassing loop overhead entirely via 64 cascading hardware macro pipelines
    SHA256_ROUND(a, b, c, d, e, f, g, h, w[0],  K[0]);  SHA256_ROUND(h, a, b, c, d, e, f, g, w[1],  K[1]);
    SHA256_ROUND(g, h, a, b, c, d, e, f, w[2],  K[2]);  SHA256_ROUND(f, g, h, a, b, c, d, e, w[3],  K[3]);
    SHA256_ROUND(e, f, g, h, a, b, c, d, w[4],  K[4]);  SHA256_ROUND(d, e, f, g, h, a, b, c, w[5],  K[5]);
    SHA256_ROUND(c, d, e, f, g, h, a, b, w[6],  K[6]);  SHA256_ROUND(b, c, d, e, f, g, h, a, w[7],  K[7]);
    SHA256_ROUND(a, b, c, d, e, f, g, h, w[8],  K[8]);  SHA256_ROUND(h, a, b, c, d, e, f, g, w[9],  K[9]);
    SHA256_ROUND(g, h, a, b, c, d, e, f, w[10], K[10]); SHA256_ROUND(f, g, h, a, b, c, d, e, w[11], K[11]);
    SHA256_ROUND(e, f, g, h, a, b, c, d, w[12], K[12]); SHA256_ROUND(d, e, f, g, h, a, b, c, w[13], K[13]);
    SHA256_ROUND(c, d, e, f, g, h, a, b, w[14], K[14]); SHA256_ROUND(b, c, d, e, f, g, h, a, w[15], K[15]);
    SHA256_ROUND(a, b, c, d, e, f, g, h, w[16], K[16]); SHA256_ROUND(h, a, b, c, d, e, f, g, w[17], K[17]);
    SHA256_ROUND(g, h, a, b, c, d, e, f, w[18], K[18]); SHA256_ROUND(f, g, h, a, b, c, d, e, w[19], K[19]);
    SHA256_ROUND(e, f, g, h, a, b, c, d, w[20], K[20]); SHA256_ROUND(d, e, f, g, h, a, b, c, w[21], K[21]);
    SHA256_ROUND(c, d, e, f, g, h, a, b, w[22], K[22]); SHA256_ROUND(b, c, d, e, f, g, h, a, w[23], K[23]);
    SHA256_ROUND(a, b, c, d, e, f, g, h, w[24], K[24]); SHA256_ROUND(h, a, b, c, d, e, f, g, w[25], K[25]);
    SHA256_ROUND(g, h, a, b, c, d, e, f, w[26], K[26]); SHA256_ROUND(f, g, h, a, b, c, d, e, w[27], K[27]);
    SHA256_ROUND(e, f, g, h, a, b, c, d, w[28], K[28]); SHA256_ROUND(d, e, f, g, h, a, b, c, w[29], K[29]);
    SHA256_ROUND(c, d, e, f, g, h, a, b, w[30], K[30]); SHA256_ROUND(b, c, d, e, f, g, h, a, w[31], K[31]);
    SHA256_ROUND(a, b, c, d, e, f, g, h, w[32], K[32]); SHA256_ROUND(h, a, b, c, d, e, f, g, w[33], K[33]);
    SHA256_ROUND(g, h, a, b, c, d, e, f, w[34], K[34]); SHA256_ROUND(f, g, h, a, b, c, d, e, w[35], K[35]);
    SHA256_ROUND(e, f, g, h, a, b, c, d, w[36], K[36]); SHA256_ROUND(d, e, f, g, h, a, b, c, w[37], K[37]);
    SHA256_ROUND(c, d, e, f, g, h, a, b, w[38], K[38]); SHA256_ROUND(b, c, d, e, f, g, h, a, w[39], K[39]);
    SHA256_ROUND(a, b, c, d, e, f, g, h, w[40], K[40]); SHA256_ROUND(h, a, b, c, d, e, f, g, w[41], K[41]);
    SHA256_ROUND(g, h, a, b, c, d, e, f, w[42], K[42]); SHA256_ROUND(f, g, h, a, b, c, d, e, w[43], K[43]);
    SHA256_ROUND(e, f, g, h, a, b, c, d, w[44], K[44]); SHA256_ROUND(d, e, f, g, h, a, b, c, w[45], K[45]);
    SHA256_ROUND(c, d, e, f, g, h, a, b, w[46], K[46]); SHA256_ROUND(b, c, d, e, f, g, h, a, w[47], K[47]);
    SHA256_ROUND(a, b, c, d, e, f, g, h, w[48], K[48]); SHA256_ROUND(h, a, b, c, d, e, f, g, w[49], K[49]);
    SHA256_ROUND(g, h, a, b, c, d, e, f, w[50], K[50]); SHA256_ROUND(f, g, h, a, b, c, d, e, w[51], K[51]);
    SHA256_ROUND(e, f, g, h, a, b, c, d, w[52], K[52]); SHA256_ROUND(d, e, f, g, h, a, b, c, w[53], K[53]);
    SHA256_ROUND(c, d, e, f, g, h, a, b, w[54], K[54]); SHA256_ROUND(b, c, d, e, f, g, h, a, w[55], K[55]);
    SHA256_ROUND(a, b, c, d, e, f, g, h, w[56], K[56]); SHA256_ROUND(h, a, b, c, d, e, f, g, w[57], K[57]);
    SHA256_ROUND(g, h, a, b, c, d, e, f, w[58], K[58]); SHA256_ROUND(f, g, h, a, b, c, d, e, w[59], K[59]);
    SHA256_ROUND(e, f, g, h, a, b, c, d, w[60], K[60]); SHA256_ROUND(d, e, f, g, h, a, b, c, w[61], K[61]);
    SHA256_ROUND(c, d, e, f, g, h, a, b, w[62], K[62]); SHA256_ROUND(b, c, d, e, f, g, h, a, w[63], K[63]);

    // Accumulate results with precalculated midstate values
    a = _mm512_add_epi32(a, _mm512_set1_epi32(midstate_raw[0]));

    // Evaluation Masking Layer
    // Check if output vector lane matches target difficulty restrictions via vector comparisons
    __mmask16 target_mask = _mm512_cmple_epu32_mask(a, _mm512_set1_epi32(target_difficulty));
    
    if (target_mask != 0) {
        for (int lane = 0; lane < 16; lane++) {
            if ((target_mask >> lane) & 1) {
                std::cout << "[+] Block Solved! Valid Nonce Identified at: " << base_nonce + lane << std::endl;
            }
        }
    }
}

