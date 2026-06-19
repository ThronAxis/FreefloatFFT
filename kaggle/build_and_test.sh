#!/bin/bash
# ============================================================
# FreefloatFFT -- Kaggle Build & Test (All-in-one)
# ============================================================
# Usage in Kaggle notebook:
#   !git clone https://github.com/ThronAxis/FreefloatFFT.git
#   !bash FreefloatFFT/kaggle/build_and_test.sh
# ============================================================

echo ""
echo "============================================================"
echo "  FreefloatFFT -- Kaggle Build & Test"
echo "============================================================"

# -- Step 1: Detect GPU -----------------------------------------------
echo ""
echo "[1/5] Detecting GPU..."
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCH="sm_${COMPUTE_CAP}"
echo "  Architecture: $ARCH"

# -- Step 2: Find source directory -------------------------------------
echo ""
echo "[2/5] Finding source directory..."

# Auto-detect: try all common locations
for DIR in \
    "$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)" \
    "/kaggle/working/FreefloatFFT" \
    "/kaggle/working/freefloatfft" \
    "/kaggle/input/FreefloatFFT" \
    "/kaggle/input/freefloatfft"; do
    if [ -f "$DIR/src/freefloatfft.cu" ]; then
        SRC_DIR="$DIR"
        break
    fi
done

if [ -z "$SRC_DIR" ]; then
    echo "  ERROR: Cannot find FreefloatFFT source directory!"
    echo "  Make sure you cloned: git clone https://github.com/ThronAxis/FreefloatFFT.git"
    exit 1
fi
echo "  Source: $SRC_DIR"

# -- Step 3: Build ----------------------------------------------------
BUILD_DIR="/kaggle/working/fft_build"
mkdir -p "$BUILD_DIR"
echo "  Build:  $BUILD_DIR"

echo ""
echo "[3/5] Compiling..."

# Shared library
echo "  -> libfreefloatfft.so"
nvcc -O3 -arch="$ARCH" \
     --use_fast_math -std=c++17 \
     -shared -Xcompiler -fPIC \
     -I"$SRC_DIR/include" \
     "$SRC_DIR/src/freefloatfft.cu" \
     -o "$BUILD_DIR/libfreefloatfft.so"

if [ $? -ne 0 ]; then
    echo "  FAILED: Library compilation error (see above)"
    exit 1
fi
echo "  OK"

# Test binary
echo "  -> test_correctness"
nvcc -O3 -arch="$ARCH" \
     --use_fast_math -std=c++17 \
     -I"$SRC_DIR/include" \
     "$SRC_DIR/tests/test_correctness.cu" \
     -o "$BUILD_DIR/test_correctness"

if [ $? -ne 0 ]; then
    echo "  FAILED: Test compilation error (see above)"
    exit 1
fi
echo "  OK"

# Benchmark binary
echo "  -> freefloatfft_bench"
nvcc -O3 -arch="$ARCH" \
     --use_fast_math -std=c++17 \
     -I"$SRC_DIR/include" \
     "$SRC_DIR/src/freefloatfft_bench.cu" \
     -o "$BUILD_DIR/freefloatfft_bench"

if [ $? -ne 0 ]; then
    echo "  FAILED: Benchmark compilation error (see above)"
    exit 1
fi
echo "  OK"

echo ""
echo "  Build artifacts:"
ls -lh "$BUILD_DIR/"

# -- Step 4: Run correctness tests ------------------------------------
echo ""
echo "[4/5] Running correctness tests..."
echo "------------------------------------------------------------"
"$BUILD_DIR/test_correctness"
TEST_EXIT=$?
echo "------------------------------------------------------------"
if [ $TEST_EXIT -ne 0 ]; then
    echo "  TESTS FAILED (exit code $TEST_EXIT)"
else
    echo "  ALL TESTS PASSED"
fi

# -- Step 5: Run benchmarks -------------------------------------------
echo ""
echo "[5/5] Running benchmarks..."
echo "------------------------------------------------------------"
"$BUILD_DIR/freefloatfft_bench"
echo "------------------------------------------------------------"

echo ""
echo "============================================================"
echo "  Build & Test Complete!"
echo ""
echo "  For Python usage:"
echo "    import sys"
echo "    sys.path.insert(0, '$SRC_DIR/python')"
echo "    from freefloatfft import FreefloatFFT"
echo "============================================================"
