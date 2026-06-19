#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# FreefloatFFT — Kaggle Build Script
# ═══════════════════════════════════════════════════════════════════════════════
#
# Run this in a Kaggle GPU notebook cell:
#   %%bash
#   bash /kaggle/working/freefloatfft/kaggle/build.sh
#
# Or inline:
#   !bash freefloatfft/kaggle/build.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            FreefloatFFT — Kaggle Build Script               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Detect GPU architecture ────────────────────────────────────────

ARCH=$(python3 -c "
import subprocess
result = subprocess.run(
    ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
    capture_output=True, text=True
)
cap = result.stdout.strip().replace('.', '')
print(f'sm_{cap}')
")

echo "  Detected GPU architecture: $ARCH"

# ── Step 2: Detect source directory ────────────────────────────────────────

# Try common locations (case-sensitive filesystem)
if [ -d "/kaggle/working/freefloatfft" ]; then
    SRC_DIR="/kaggle/working/freefloatfft"
elif [ -d "/kaggle/working/FreefloatFFT" ]; then
    SRC_DIR="/kaggle/working/FreefloatFFT"
elif [ -d "/kaggle/input/freefloatfft" ]; then
    SRC_DIR="/kaggle/input/freefloatfft"
elif [ -d "/kaggle/input/FreefloatFFT" ]; then
    SRC_DIR="/kaggle/input/FreefloatFFT"
else
    SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

echo "  Source directory: $SRC_DIR"

# ── Step 3: Create build directory ─────────────────────────────────────────

BUILD_DIR="/kaggle/working/fft_build"
mkdir -p "$BUILD_DIR"
echo "  Build directory: $BUILD_DIR"

# ── Step 4: Compile shared library ─────────────────────────────────────────

echo ""
echo "  Compiling shared library..."

nvcc -O3 -arch="$ARCH" \
     -Xptxas="-v,-warn-lmem-usage,-warn-spills" \
     --use_fast_math \
     -std=c++17 \
     -shared -Xcompiler -fPIC \
     -I"$SRC_DIR/include" \
     "$SRC_DIR/src/freefloatfft.cu" \
     -o "$BUILD_DIR/libfreefloatfft.so" 2>&1

echo "  ✓ Library built: $BUILD_DIR/libfreefloatfft.so"

# ── Step 5: Compile test binary ────────────────────────────────────────────

echo "  Compiling test binary..."

nvcc -O3 -arch="$ARCH" \
     --use_fast_math \
     -std=c++17 \
     -I"$SRC_DIR/include" \
     "$SRC_DIR/tests/test_correctness.cu" \
     -o "$BUILD_DIR/test_correctness" 2>&1

echo "  ✓ Tests built: $BUILD_DIR/test_correctness"

# ── Step 6: Compile benchmark binary ──────────────────────────────────────

echo "  Compiling benchmark binary..."

nvcc -O3 -arch="$ARCH" \
     --use_fast_math \
     -std=c++17 \
     -I"$SRC_DIR/include" \
     "$SRC_DIR/src/freefloatfft_bench.cu" \
     -o "$BUILD_DIR/freefloatfft_bench" 2>&1

echo "  ✓ Benchmark built: $BUILD_DIR/freefloatfft_bench"

# ── Step 7: Summary ───────────────────────────────────────────────────────

echo ""
echo "  Build artifacts:"
ls -lh "$BUILD_DIR/"
echo ""
echo "  Build complete! ✓"
echo ""
echo "  Usage in Python:"
echo "    import sys"
echo "    sys.path.insert(0, '$SRC_DIR/python')"
echo "    from freefloatfft import FreefloatFFT"
echo ""
echo "  Run tests:"
echo "    !$BUILD_DIR/test_correctness"
echo ""
echo "  Run benchmarks:"
echo "    !$BUILD_DIR/freefloatfft_bench"
echo ""
