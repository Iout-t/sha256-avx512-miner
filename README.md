# High-Throughput AVX-512 Vectorized SHA-256 Miner

Experimental low-overhead SHA-256 pipeline layout implementation utilizing vectorized modern CPU capabilities to overcome common memory and block evaluation software limits.

## Implemented Bottleneck Patches

1. **AVX-512 Lane Widening**: Maps 16 nonces across 512-bit vector zones (`__m512i`) to eliminate independent iteration requirements.
2. **Loop Flattener (ILP Optimization)**: Implements 64 fully unrolled macro evaluation fields to completely avoid CPU internal pipeline branch penalties.
3. **Hardware Midstate Isolation**: Evaluates chunk-1 logic independently once per 4GB cycle range, cutting the computational weight of the initial SHA-256 pass by 50%.
4. **Asynchronous Memory Line Syncing**: Issues active hardware line hints (`_mm_prefetch`) to draw next-step execution packages directly into L1 caches ahead of consumption intervals.

## Build Requirements

* Compiler with native AVX-512 capabilities (GCC 9+, Clang 10+, or MSVC 2019+)
* CMake v3.15 or newer
* Compatible Intel or AMD CPU with `AVX512F` and `AVX512VL` flags supported

## Installation and Execution

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./sha256_miner
