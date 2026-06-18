# FreefloatFFT

> **High-Performance Custom CUDA FFT for Small Batched Signals**

A from-scratch CUDA Fast Fourier Transform library optimized for the regime cuFFT ignores: **small-to-medium signal lengths** (N = 64 to 1024) processed in **very large batches** (B ≥ 10,000).

---

## ⚡ Performance Highlights

| Metric | FreefloatFFT | cuFFT |
|--------|-------------|-------|
| **Plan overhead** | 0 (one-time precompute) | 50–500 µs per call |
| **Kernel launches per batch** | 1 | 2–4 |
| **Shared memory bank conflicts** | 0 (proven) | Present |
| **Warp barriers (N=512)** | 4 (cross-warp only) | 9 (all stages) |
| **Target speedup** | **3–8×** over cuFFT | Baseline |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HOST API (C++ / Python)                    │
│  plan_create(N, B) → execute(d_in, d_out, dir) → destroy()  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                  CUDA KERNEL LIBRARY                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Warp Shuffle │  │ Shared Memory│  │ Template Dispatch │  │
│  │ Stages (0-4) │  │ Stages (5+)  │  │ N={64..1024}     │  │
│  │ No barriers! │  │ Dual-array   │  │ Full unrolling   │  │
│  └──────────────┘  │ Zero conflict│  └───────────────────┘  │
│                    └──────────────┘                           │
└──────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
freefloatfft/
├── include/
│   ├── freefloatfft.h              # Public C++/C API
│   ├── freefloatfft_math.cuh       # Complex arithmetic (cadd, cmul, butterfly)
│   ├── freefloatfft_constants.cuh  # Constant memory (twiddle, bitrev)
│   ├── freefloatfft_warp.cuh       # Warp-shuffle butterfly stages
│   ├── freefloatfft_smem.cuh       # Shared memory butterfly stages
│   └── freefloatfft_kernel.cuh     # Main kernel template + launcher
├── src/
│   ├── freefloatfft.cu             # Plan management + C API exports
│   └── freefloatfft_bench.cu       # Benchmark suite
├── python/
│   ├── freefloatfft.py             # Python ctypes wrapper
│   └── validate.py                 # scipy.fft comparison
├── kaggle/
│   └── build.sh                    # Kaggle GPU compilation script
├── tests/
│   ├── test_correctness.cu         # CUDA unit tests
│   └── test_bank_conflicts.py      # Bank conflict verification
├── Makefile                        # Build system
└── README.md                       # This file
```

## 🚀 Quick Start

### Build (Linux / Kaggle)

```bash
# Build everything (library + tests + benchmarks)
make all ARCH=sm_75    # T4
# make all ARCH=sm_60  # P100
# make all ARCH=sm_80  # A100

# Run correctness tests
make run_test

# Run benchmarks
make run_bench
```

### Kaggle Notebook

```python
# Cell 1: Build
%%bash
bash freefloatfft/kaggle/build.sh

# Cell 2: Use
import sys
sys.path.insert(0, '/kaggle/working/freefloatfft/python')
from freefloatfft import FreefloatFFT
import cupy as cp

engine = FreefloatFFT(N=512, B=100000)
x = (cp.random.randn(100000, 512) + 1j * cp.random.randn(100000, 512)).astype(cp.complex64)
X = engine.forward(x)
```

### C++ API

```cpp
#include "freefloatfft.h"

fft::Plan plan;
fft::plan_create(plan, 512, 100000);                        // N=512, B=100K
fft::plan_execute(plan, d_in, d_out, fft::Direction::FORWARD);
fft::plan_destroy(plan);
```

## 🔬 Key Optimizations

### 1. Warp-Synchronous Butterfly Stages
First `log₂(32) = 5` stages use `__shfl_xor_sync` — **zero shared memory traffic, zero `__syncthreads()` barriers**.

### 2. Bank-Conflict-Free Shared Memory
Dual-array separation with +1 `float2` offset guarantees **zero bank conflicts** for all butterfly strides. Analytically proven for all N ∈ {64, 128, 256, 512, 1024}.

### 3. Precomputed Twiddle Factors
Double-precision trigonometric recurrence, stored as `float32` in `__constant__` memory. L1 cache hit rate > 95%.

### 4. Template-Specialized Kernels
Each N value gets a fully-unrolled, register-optimized kernel via C++ templates. Zero runtime branching.

## 📊 Supported Configurations

| Parameter | Values |
|-----------|--------|
| **N** (transform length) | 64, 128, 256, 512, 1024 |
| **B** (batch size) | 1 to 10,000,000 |
| **Precision** | float32 (complex64) |
| **Direction** | Forward / Inverse (with 1/N normalization) |
| **In-place** | Supported (d_in == d_out) |

## 🛠️ Requirements

- **CUDA Toolkit** ≥ 11.0
- **C++17** compiler
- **GPU**: NVIDIA with compute capability ≥ 6.0
- **Python** ≥ 3.8 (for Python wrapper)
- **CuPy** (for GPU array support in Python)

## 📄 Documentation

- [Product Requirements (PRD)](FreefloatFFT_PRD.md)
- [Technical Design Document](FreefloatFFT_DESIGN.md)

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*FreefloatFFT — Built from first principles for the regime cuFFT ignores.*
*CUDA ML Systems Engineering | Antigravity Field Computing Division*
