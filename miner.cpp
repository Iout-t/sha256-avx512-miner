#include "miner.h"

alignas(64) const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROR32(x, n) _mm512_or_si512(_mm512_srli_epi32(x, n), _mm512_slli_epi32(x, 32 - n))

// Ternary Logic Truth Tables
#define CH_TERNARY(x, y, z)  _mm512_ternarylogic_epi32(x, y, z, 0xCA)
#define MAJ_TERNARY(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0xE8)

#define EP0(x) _mm512_xor_si512(ROR32(x, 2), _mm512_xor_si512(ROR32(x, 13), ROR32(x, 22)))
#define EP1(x) _mm512_xor_si512(ROR32(x, 6), _mm512_xor_si512(ROR32(x, 11), ROR32(x, 25)))
#define SIG0(x) _mm512_xor_si512(ROR32(x, 7), _mm512_xor_si512(ROR32(x, 18), _mm512_srli_epi32(x, 3)))
#define SIG1(x) _mm512_xor_si512(ROR32(x, 17), _mm512_xor_si512(ROR32(x, 19), _mm512_srli_epi32(x, 10)))

#define SHA256_ROUND_TERNARY(a, b, c, d, e, f, g, h, w_val, k_val) { \
    __m512i ch_res  = CH_TERNARY(e, f, g); \
    __m512i maj_res = MAJ_TERNARY(a, b, c); \
    __m512i ep1_res = EP1(e); \
    __m512i ep0_res = EP0(a); \
    __m512i t1 = _mm512_add_epi32(h, _mm512_add_epi32(ep1_res, _mm512_add_epi32(ch_res, _mm512_add_epi32(w_val, _mm512_set1_epi32(k_val))))); \
    __m512i t2 = _mm512_add_epi32(ep0_res, maj_res); \
    d = _mm512_add_epi32(d, t1); \
    h = _mm512_add_epi32(t1, t2); \
}

#define TRANSPOSE_INTERLEAVE_32(r0, r1, t0, t1) { \
    t0 = _mm512_unpacklo_epi32(r0, r1); \
    t1 = _mm512_unpackhi_epi32(r0, r1); \
}

#define TRANSPOSE_INTERLEAVE_64(r0, r1, t0, t1) { \
    t0 = _mm512_unpacklo_epi64(r0, r1); \
    t1 = _mm512_unpackhi_epi64(r0, r1); \
}

[[gnu::always_inline]] 
void execute_max_boundary_core(
    const uint32_t* midstate_raw,
    const CryptoBlock* multi_block_buffer,
    uint32_t target_difficulty
) {
    __m512i r0 = _mm512_loadu_si512((__m512i*)&multi_block_buffer[0]);
    __m512i r1 = _mm512_loadu_si512((__m512i*)&multi_block_buffer[1]);
    __m512i r2 = _mm512_loadu_si512((__m512i*)&multi_block_buffer[2]);
    __m512i r3 = _mm512_loadu_si512((__m512i*)&multi_block_buffer[3]);
    
    __m512i w[16];
    __m512i t0, t1, t2, t3;
    
    TRANSPOSE_INTERLEAVE_32(r0, r1, t0, t1);
    TRANSPOSE_INTERLEAVE_32(r2, r3, t2, t3);
    TRANSPOSE_INTERLEAVE_64(t0, t2, w[0], w[1]);
    TRANSPOSE_INTERLEAVE_64(t1, t3, w[2], w[3]);

    w[4] = _mm512_set1_epi32(0x80000000); 
    w[5] = _mm512_setzero_si512(); w[6] = _mm512_setzero_si512(); w[7] = _mm512_setzero_si512();
    w[8] = _mm512_setzero_si512(); w[9] = _mm512_setzero_si512(); w[10] = _mm512_setzero_si512();
    w[11] = _mm512_setzero_si512(); w[12] = _mm512_setzero_si512(); w[13] = _mm512_setzero_si512();
    w[14] = _mm512_setzero_si512();
    w[15] = _mm512_set1_epi32(0x00000280);

    __m512i a = _mm512_set1_epi32(midstate_raw[0]); __m512i b = _mm512_set1_epi32(midstate_raw[1]);
    __m512i c = _mm512_set1_epi32(midstate_raw[2]); __m512i d = _mm512_set1_epi32(midstate_raw[3]);
    __m512i e = _mm512_set1_epi32(midstate_raw[4]); __m512i f = _mm512_set1_epi32(midstate_raw[5]);
    __m512i g = _mm512_set1_epi32(midstate_raw[6]); __m512i h = _mm512_set1_epi32(midstate_raw[7]);

    SHA256_ROUND_TERNARY(a, b, c, d, e, f, g, h, w[0], K[0]);  
    SHA256_ROUND_TERNARY(h, a, b, c, d, e, f, g, w[1], K[1]);
    SHA256_ROUND_TERNARY(g, h, a, b, c, d, e, f, w[2], K[2]);  
    SHA256_ROUND_TERNARY(f, g, h, a, b, c, d, e, w[3], K[3]);

    SHA256_ROUND_TERNARY(e, f, g, h, a, b, c, d, w[4],  K[4]);  SHA256_ROUND_TERNARY(d, e, f, g, h, a, b, c, w[5],  K[5]);
    SHA256_ROUND_TERNARY(c, d, e, f, g, h, a, b, w[6],  K[6]);  SHA256_ROUND_TERNARY(b, c, d, e, f, g, h, a, w[7],  K[7]);
    SHA256_ROUND_TERNARY(a, b, c, d, e, f, g, h, w[8],  K[8]);  SHA256_ROUND_TERNARY(h, a, b, c, d, e, f, g, w[9],  K[9]);
    SHA256_ROUND_TERNARY(g, h, a, b, c, d, e, f, w[10], K[10]); SHA256_ROUND_TERNARY(f, g, h, a, b, c, d, e, w[11], K[11]);
    SHA256_ROUND_TERNARY(e, f, g, h, a, b, c, d, w[12], K[12]); SHA256_ROUND_TERNARY(d, e, f, g, h, a, b, c, w[13], K[13]);
    SHA256_ROUND_TERNARY(c, d, e, f, g, h, a, b, w[14], K[14]); SHA256_ROUND_TERNARY(b, c, d, e, f, g, h, a, w[15], K[15]);

    #pragma GCC unroll 6
    for (int t = 16; t < 64; t += 8) {
        w[t & 15] = _mm512_add_epi32(w[(t-16) & 15], _mm512_add_epi32(SIG0(w[(t-15) & 15]), _mm512_add_epi32(w[(t-7) & 15], SIG1(w[(t-2) & 15]))));
        SHA256_ROUND_TERNARY(a, b, c, d, e, f, g, h, w[t & 15], K[t]);

        w[(t+1) & 15] = _mm512_add_epi32(w[(t-15) & 15], _mm512_add_epi32(SIG0(w[t & 15]), _mm512_add_epi32(w[(t-6) & 15], SIG1(w[(t-1) & 15]))));
        SHA256_ROUND_TERNARY(h, a, b, c, d, e, f, g, w[(t+1) & 15], K[t+1]);

        w[(t+2) & 15] = _mm512_add_epi32(w[(t-14) & 15], _mm512_add_epi32(SIG0(w[(t+1) & 15]), _mm512_add_epi32(w[(t-5) & 15], SIG1(w[t & 15]))));
        SHA256_ROUND_TERNARY(g, h, a, b, c, d, e, f, w[(t+2) & 15], K[t+2]);

        w[(t+3) & 15] = _mm512_add_epi32(w[(t-13) & 15], _mm512_add_epi32(SIG0(w[(t+2) & 15]), _mm512_add_epi32(w[(t-4) & 15], SIG1(w[(t+1) & 15]))));
        SHA256_ROUND_TERNARY(f, g, h, a, b, c, d, e, w[(t+3) & 15], K[t+3]);

        w[(t+4) & 15] = _mm512_add_epi32(w[(t-12) & 15], _mm512_add_epi32(SIG0(w[(t+3) & 15]), _mm512_add_epi32(w[(t-3) & 15], SIG1(w[(t+2) & 15]))));
        SHA256_ROUND_TERNARY(e, f, g, h, a, b, c, d, w[(t+4) & 15], K[t+4]);

        w[(t+5) & 15] = _mm512_add_epi32(w[(t-11) & 15], _mm512_add_epi32(SIG0(w[(t+4) & 15]), _mm512_add_epi32(w[(t-2) & 15], SIG1(w[(t+3) & 15]))));
        SHA256_ROUND_TERNARY(d, e, f, g, h, a, b, c, w[(t+5) & 15], K[t+5]);

        w[(t+6) & 15] = _mm512_add_epi32(w[(t-10) & 15], _mm512_add_epi32(SIG0(w[(t+5) & 15]), _mm512_add_epi32(w[(t-1) & 15], SIG1(w[(t+4) & 15]))));
        SHA256_ROUND_TERNARY(c, d, e, f, g, h, a, b, w[(t+6) & 15], K[t+6]);

        w[(t+7) & 15] = _mm512_add_epi32(w[(t-9) & 15], _mm512_add_epi32(SIG0(w[(t+6) & 15]), _mm512_add_epi32(w[t & 15], SIG1(w[(t+5) & 15]))));
        SHA256_ROUND_TERNARY(b, c, d, e, f, g, h, a, w[(t+7) & 15], K[t+7]);
    }

    a = _mm512_add_epi32(a, _mm512_set1_epi32(midstate_raw[0]));

    __mmask16 match_mask = _mm512_cmple_epu32_mask(a, _mm512_set1_epi32(target_difficulty));
    if (__builtin_expect(match_mask != 0, 0)) {
        for (int lane = 0; lane < 16; lane++) {
            if ((match_mask >> lane) & 1) {
                std::cout << "[MAX] Absolute ASIC-Level Match in Lane: " << lane << std::endl;
            }
        }
    }
}
