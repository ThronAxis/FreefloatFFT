#!/bin/bash
# FreefloatFFT - Debug Build for Kaggle
# Run: !bash FreefloatFFT/kaggle/debug_build.sh

set -x  # Print every command before execution

# Find the source
SRC=""
for D in /kaggle/working/FreefloatFFT /kaggle/working/freefloatfft; do
    if [ -f "$D/src/freefloatfft.cu" ]; then SRC="$D"; break; fi
done
if [ -z "$SRC" ]; then
    SRC="$(cd "$(dirname "$0")/.." && pwd)"
fi

echo "SRC_DIR=$SRC"
ls "$SRC/src/" "$SRC/include/" "$SRC/tests/"

# Detect arch
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCH="sm_${ARCH}"
echo "ARCH=$ARCH"

mkdir -p /kaggle/working/fft_build

# Step 1: Compile library
echo ""
echo "=== COMPILING LIBRARY ==="
nvcc -O3 -arch="$ARCH" --use_fast_math -std=c++17 \
     -shared -Xcompiler -fPIC \
     -I"$SRC/include" \
     "$SRC/src/freefloatfft.cu" \
     -o /kaggle/working/fft_build/libfreefloatfft.so
echo "Library exit code: $?"

# Step 2: Compile tests
echo ""
echo "=== COMPILING TESTS ==="
nvcc -O3 -arch="$ARCH" --use_fast_math -std=c++17 \
     -I"$SRC/include" \
     "$SRC/tests/test_correctness.cu" \
     -o /kaggle/working/fft_build/test_correctness
echo "Test exit code: $?"

# Step 3: Compile bench
echo ""
echo "=== COMPILING BENCH ==="
nvcc -O3 -arch="$ARCH" --use_fast_math -std=c++17 \
     -I"$SRC/include" \
     "$SRC/src/freefloatfft_bench.cu" \
     -o /kaggle/working/fft_build/freefloatfft_bench
echo "Bench exit code: $?"

echo ""
echo "=== BUILD RESULTS ==="
ls -la /kaggle/working/fft_build/
