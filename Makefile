# ═══════════════════════════════════════════════════════════════════════════════
# FreefloatFFT — Makefile
# ═══════════════════════════════════════════════════════════════════════════════
#
# Targets:
#   make lib       — Build shared library (libfreefloatfft.so)
#   make bench     — Build benchmark binary
#   make test      — Build and run correctness tests
#   make all       — Build everything
#   make clean     — Remove build artifacts
#
# Configuration:
#   ARCH=sm_75     — Target GPU architecture (default: sm_75 for T4)
#   CUDA_HOME      — CUDA toolkit path (auto-detected)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Compiler Configuration ───────────────────────────────────────────────────

NVCC        ?= nvcc
ARCH        ?= sm_75
CUDA_HOME   ?= /usr/local/cuda

# Compiler flags
NVCC_FLAGS  := -O3 \
               -arch=$(ARCH) \
               --use_fast_math \
               -std=c++17 \
               -Xptxas="-v,-warn-lmem-usage,-warn-spills" \
               -lineinfo

# Shared library flags (Linux/Kaggle)
SO_FLAGS    := -shared -Xcompiler -fPIC

# Include paths
INCLUDES    := -I./include

# ── Directories ──────────────────────────────────────────────────────────────

BUILD_DIR   := build
SRC_DIR     := src
INC_DIR     := include
TEST_DIR    := tests

# ── Source Files ─────────────────────────────────────────────────────────────

LIB_SRC     := $(SRC_DIR)/freefloatfft.cu
BENCH_SRC   := $(SRC_DIR)/freefloatfft_bench.cu
TEST_SRC    := $(TEST_DIR)/test_correctness.cu

# ── Output Files ─────────────────────────────────────────────────────────────

LIB_OUT     := $(BUILD_DIR)/libfreefloatfft.so
BENCH_OUT   := $(BUILD_DIR)/freefloatfft_bench
TEST_OUT    := $(BUILD_DIR)/test_correctness

# ── Header Dependencies ─────────────────────────────────────────────────────

HEADERS     := $(INC_DIR)/freefloatfft.h \
               $(INC_DIR)/freefloatfft_math.cuh \
               $(INC_DIR)/freefloatfft_constants.cuh \
               $(INC_DIR)/freefloatfft_warp.cuh \
               $(INC_DIR)/freefloatfft_smem.cuh \
               $(INC_DIR)/freefloatfft_kernel.cuh

# ═══════════════════════════════════════════════════════════════════════════════
# Targets
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all lib bench test run_test run_bench clean help

all: lib bench test

# ── Shared Library ───────────────────────────────────────────────────────────

lib: $(LIB_OUT)

$(LIB_OUT): $(LIB_SRC) $(HEADERS) | $(BUILD_DIR)
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║  Building FreefloatFFT shared library                       ║"
	@echo "║  Target: $(ARCH)                                            ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	$(NVCC) $(NVCC_FLAGS) $(SO_FLAGS) $(INCLUDES) $(LIB_SRC) -o $(LIB_OUT)
	@echo ""
	@echo "  ✓ Built: $(LIB_OUT)"
	@ls -lh $(LIB_OUT)
	@echo ""

# ── Benchmark Binary ────────────────────────────────────────────────────────

bench: $(BENCH_OUT)

$(BENCH_OUT): $(BENCH_SRC) $(HEADERS) | $(BUILD_DIR)
	@echo ""
	@echo "  Building benchmark binary..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(BENCH_SRC) -o $(BENCH_OUT)
	@echo "  ✓ Built: $(BENCH_OUT)"
	@echo ""

# ── Test Binary ──────────────────────────────────────────────────────────────

test: $(TEST_OUT)

$(TEST_OUT): $(TEST_SRC) $(HEADERS) | $(BUILD_DIR)
	@echo ""
	@echo "  Building test binary..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(TEST_SRC) -o $(TEST_OUT)
	@echo "  ✓ Built: $(TEST_OUT)"
	@echo ""

# ── Run Targets ──────────────────────────────────────────────────────────────

run_test: $(TEST_OUT)
	@echo ""
	./$(TEST_OUT)

run_bench: $(BENCH_OUT)
	@echo ""
	./$(BENCH_OUT)

# ── Build Directory ──────────────────────────────────────────────────────────

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	@echo "  Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	@echo "  ✓ Clean complete"

# ── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  FreefloatFFT Build System"
	@echo "  ─────────────────────────"
	@echo "  make lib        Build shared library (libfreefloatfft.so)"
	@echo "  make bench      Build benchmark binary"
	@echo "  make test       Build test binary"
	@echo "  make all        Build everything"
	@echo "  make run_test   Build and run correctness tests"
	@echo "  make run_bench  Build and run benchmarks"
	@echo "  make clean      Remove build artifacts"
	@echo ""
	@echo "  Configuration:"
	@echo "    ARCH=sm_XX    GPU architecture (default: sm_75 for T4)"
	@echo "                  T4=sm_75, P100=sm_60, A100=sm_80, H100=sm_90"
	@echo ""
