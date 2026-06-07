#!/usr/bin/env bash
set -e
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
echo "------------------------------------------------"
echo "[+] Build Complete! Execute using: ./build/sha256_max_miner"
echo "------------------------------------------------"
