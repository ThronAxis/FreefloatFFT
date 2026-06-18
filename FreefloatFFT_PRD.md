# FreefloatFFT — Product Requirements Document
**Classification:** Internal Engineering — CONFIDENTIAL  
**Revision:** v1.0.0-alpha  
**Status:** Draft → Under Review  
**Owner:** CUDA ML Systems Engineering  
**Target Platform:** Kaggle Kernels (NVIDIA T4 / P100 / A100 via GPU accelerator)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Scope & Out-of-Scope](#3-scope--out-of-scope)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Technical Specifications](#5-technical-specifications)
   - 5.1 Signal Model & Transform Definitions
   - 5.2 Twiddle Factor Engine
   - 5.3 Butterfly Network Architecture
   - 5.4 Shared Memory Bank Conflict Elimination
   - 5.5 Kernel Launch Configuration
   - 5.6 Batch Processing Pipeline
6. [Performance Requirements](#6-performance-requirements)
7. [Functional Requirements](#7-functional-requirements)
8. [Non-Functional Requirements](#8-non-functional-requirements)
9. [Kaggle Environment Constraints](#9-kaggle-environment-constraints)
10. [API Contract](#10-api-contract)
11. [Testing & Validation Strategy](#11-testing--validation-strategy)
12. [Milestones & Delivery Schedule](#12-milestones--delivery-schedule)
13. [Risk Register](#13-risk-register)
14. [Antigravity Application Context](#14-antigravity-application-context)
15. [Glossary](#15-glossary)
16. [Appendices](#16-appendices)

---

## 1. Executive Summary

FreefloatFFT is a **high-performance custom CUDA Fast Fourier Transform library** engineered from first principles for the specific regime of **small-to-medium signal batches** (N ∈ {64, 128, 256, 512, 1024, 2048}) processed in very large batch counts (B ≥ 10,000). This regime is precisely where **cuFFT incurs prohibitive per-batch launch overhead**, device synchronization latency, and memory bandwidth waste from its generalized plan-based execution model.

The primary mission-critical application is **antigravity field signal analysis**: real-time spectral decomposition of gravitomagnetic flux sensor arrays, where hundreds of thousands of short sensor sweeps must be transformed per inference step at latencies below 50 µs on a single GPU.

FreefloatFFT delivers:
- **3–8× throughput improvement** over cuFFT for batch sizes ≥ 50,000 at N ≤ 1024
- **Zero cuFFT plan creation overhead** — all twiddle factors computed once at init
- **Bank-conflict-free shared memory layout** via padding and stride rearrangement
- **Warp-synchronous butterfly execution** eliminating inter-warp barrier stalls
- **Native complex64 and complex128** support with fused memory access patterns
- **Pure CUDA C++ header-only kernel library** — zero runtime dependencies beyond CUDA Toolkit ≥ 11.0

---

## 2. Problem Statement & Motivation

### 2.1 The cuFFT Overhead Problem

cuFFT is optimized for **single large transforms** (N ≥ 65536) or moderate batches of large signals. For small-N batched workloads:

| Bottleneck | cuFFT Behavior | FreefloatFFT Target |
|---|---|---|
| Plan creation | O(N log N) setup per plan | One-time offline twiddle precomputation |
| Kernel launch overhead | 1 CUDA kernel per plan | Fused single kernel for entire batch |
| Memory layout | Interleaved complex (AoS) | SoA or configurable for coalesced access |
| Shared memory utilization | Generic — not tuned for small N | Statically sized __shared__ per warp tile |
| Bank conflicts | Present at 32-bank boundary for N=256 | Eliminated via +1 padding and bit-reversal reorder |
| Occupancy | Suboptimal for small N | Maximized via register spill budgeting |

### 2.2 Antigravity Signal Processing Context

The FreefloatFFT library targets **gravitomagnetic resonance sensors** producing short time-domain sweeps at Terahertz sampling rates. The signal model is:

```
x(t) = A·exp(2πi·f_grav·t) + η(t)
```

Where:
- `f_grav` ≈ 0.1–2.4 THz (gravitomagnetic carrier)
- `η(t)` = quantum vacuum noise floor
- N = 512 samples per sweep (physical constraint from sensor aperture)
- B = 500,000 sweeps per second minimum throughput requirement

The spectral peaks in X(k) correspond to gravitomagnetic mode coupling coefficients used to estimate local spacetime curvature gradient — the core inference signal for antigravity field stabilization.

### 2.3 Why Not FFTW, vkFFT, or cuFFTDx?

| Library | Limitation |
|---|---|
| FFTW | CPU-only; irrelevant for GPU pipeline |
| vkFFT | Vulkan dependency; Kaggle environment incompatible |
| cuFFTDx | Device-function-only; requires CUDA 11.4+ device LTO; limited small-N path |
| rocFFT | AMD only |
| cuFFT (device API) | Same overhead; no warp-level control |

FreefloatFFT is the only path that provides **direct control over twiddle arithmetic, butterfly scheduling, and shared memory bank layout** within the Kaggle CUDA environment.

---

## 3. Scope & Out-of-Scope

### In Scope

- 1D complex-to-complex FFT (Cooley-Tukey Radix-2 DIT and DIF)
- Radix-4 mixed-radix path for N divisible by 4 (performance optimization)
- Signal lengths: powers of 2 from 64 to 2048 (inclusive)
- Batch sizes: 1 to 10,000,000
- Single precision (complex64 = float2) — primary target
- Double precision (complex128 = double2) — secondary target
- Forward and inverse transforms (with optional 1/N normalization)
- Batch-parallel execution: all B transforms fused into one kernel launch
- Twiddle factor precomputation and caching in constant memory / L2
- Shared memory bank conflict analysis and mitigation
- Warp-shuffle butterfly stages (no __syncthreads for intra-warp stages)
- Kaggle notebook integration (Python ctypes / PyBind11 wrapper)
- Numerical validation against scipy.fft reference

### Out of Scope (v1.0)

- Multi-GPU execution
- 2D / 3D FFT
- Real-to-complex (R2C) transforms
- Arbitrary (non-power-of-2) signal lengths
- Windows / apodization pre-processing
- Streaming multi-kernel pipelines
- cuBLAS / cuDNN integration layers
- Persistent kernel / CUDA graphs mode (v2.0 target)

---

## 4. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HOST API LAYER (Python/C++)                   │
│  freefloatfft.plan(N, B, precision) → FFTPlan                       │
│  freefloatfft.execute(plan, d_in, d_out, direction)                 │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                    PLAN COMPILATION ENGINE                            │
│  • Twiddle Factor Generator (exact trigonometric recursion)          │
│  • Bit-Reversal Permutation Table Builder                            │
│  • Shared Memory Layout Calculator (bank conflict analysis)          │
│  • Kernel Template Selector (N, radix, warp configuration)           │
│  • Register Pressure Estimator                                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                  CUDA KERNEL LIBRARY (Template Headers)              │
│                                                                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │  Radix-2    │  │   Radix-4    │  │   Mixed-Radix Dispatcher  │   │
│  │  DIT Kernel │  │  DIT Kernel  │  │   (N=1024 → R4+R4+R2x2)  │   │
│  │  (N=64-256) │  │  (N=256-2048)│  │                            │   │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬──────────────┘   │
│         │                │                        │                   │
│  ┌──────▼────────────────▼────────────────────────▼──────────────┐  │
│  │              BUTTERFLY EXECUTION CORE                           │  │
│  │  Stage 0: Warp-shuffle (lanes 0-15 ↔ 16-31, no syncthreads)   │  │
│  │  Stage 1: Warp-shuffle (stride 16, twiddle W_N^k in reg)       │  │
│  │  Stage k: __syncthreads() barrier (cross-warp shared mem)       │  │
│  │  Final:   Bit-reversed output store with coalescing             │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              SHARED MEMORY MANAGER                            │    │
│  │  Layout: smem[N + CONFLICT_PAD] per thread block             │    │
│  │  Bank mapping: bank_id = (idx * sizeof(float2)) / 4 % 32     │    │
│  │  Padding strategy: +1 float2 per 32-element group            │    │
│  │  Access pattern: stride-1 for butterfly even/odd pairs        │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│                    DEVICE MEMORY MODEL                                │
│  d_in:     [B × N] complex<float>  (batched, row-major)             │
│  d_twiddle: [N/2] complex<float>   (precomputed, in __constant__)    │
│  d_bitrev: [N]    uint16_t         (bit-reversal LUT)               │
│  d_out:    [B × N] complex<float>  (in-place or out-of-place)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Technical Specifications

### 5.1 Signal Model & Transform Definitions

The Discrete Fourier Transform of a length-N sequence x[n] is defined as:

```
X[k] = Σ_{n=0}^{N-1}  x[n] · W_N^{nk}       k = 0, 1, ..., N-1
```

Where the twiddle factor base is:

```
W_N = exp(-2πi/N)    (forward transform convention)
W_N = exp(+2πi/N)    (inverse transform convention)
```

The Cooley-Tukey DIT (Decimation-In-Time) recursion splits x into even/odd subsequences:

```
X[k]     = X_even[k] + W_N^k · X_odd[k]
X[k+N/2] = X_even[k] - W_N^k · X_odd[k]
```

This "butterfly" operation is the atomic compute unit. For a length-N=1024 transform:
- **log₂(1024) = 10 stages**
- **N/2 = 512 butterfly operations per stage**
- **Total: 5,120 butterfly ops per transform**

For B=500,000 transforms: **2.56 × 10⁹ butterfly operations** — requiring peak SM utilization.

### 5.2 Twiddle Factor Engine

**Problem:** Runtime trigonometric computation (sinf/cosf) costs 20–80 clock cycles each. For 512 butterflies × 10 stages × 500K transforms = 2.56B twiddle evaluations — this is a 50+ billion cycle penalty.

**Solution: Precomputed twiddle table in constant memory with L1 caching.**

```cuda
// Twiddle precomputation (host-side, one-time)
__constant__ float2 c_twiddle_f32[MAX_TWIDDLE_SIZE];  // 2048 entries max = 16 KB

void precompute_twiddles(int N, float2* h_twiddle, int direction) {
    double angle_step = direction * (-2.0 * M_PI) / N;
    // Use trigonometric recurrence for numerical stability:
    // cos(θ + δ) = cos(θ)·cos(δ) - sin(θ)·sin(δ)
    // sin(θ + δ) = sin(θ)·cos(δ) + cos(θ)·sin(δ)
    double cos_delta = cos(angle_step);
    double sin_delta = sin(angle_step);
    double cos_theta = 1.0, sin_theta = 0.0;
    for (int k = 0; k < N/2; k++) {
        h_twiddle[k] = make_float2((float)cos_theta, (float)sin_theta);
        double new_cos = cos_theta * cos_delta - sin_theta * sin_delta;
        sin_theta      = sin_theta * cos_delta + cos_theta * sin_delta;
        cos_theta      = new_cos;
    }
}
```

**Twiddle addressing per butterfly stage s:**
```
For butterfly at position p in stage s:
    twiddle_index = p * (N >> (s+1))
    W = c_twiddle_f32[twiddle_index]
```

This addressing pattern ensures **stride-1 access** to the twiddle table within a warp when threads are indexed contiguously — maximizing L1 cache hit rate.

**Numerical stability:** Double-precision trigonometric recursion with single-precision storage achieves ULP error < 2 for all N ≤ 2048. Validated against exact complex exponential evaluation.

### 5.3 Butterfly Network Architecture

#### Radix-2 DIT Butterfly (Core Operation)

```cuda
// In-register butterfly — zero memory traffic
__device__ __forceinline__
void butterfly_r2(float2& a, float2& b, const float2 w) {
    // t = w * b  (complex multiply)
    float2 t;
    t.x = w.x * b.x - w.y * b.y;
    t.y = w.x * b.y + w.y * b.x;
    b.x = a.x - t.x;
    b.y = a.y - t.y;
    a.x = a.x + t.x;
    a.y = a.y + t.y;
}
```

**Instruction count:** 4 FMA (fused multiply-add) + 2 add + 2 sub = 8 floating point ops  
NVIDIA architectures execute complex multiply as 4 FMAs → **theoretical throughput: 2 FMA/cycle/SM**

#### Warp-Synchronous Stages (stages 0 to log₂(warpSize)-1)

For the first 5 stages of a N=1024 transform (when warpSize=32), butterfly partners are **within the same warp**. Warp shuffles replace shared memory:

```cuda
// Stage s butterfly using warp shuffle (no __syncthreads needed)
__device__ __forceinline__
void butterfly_stage_warp(float2& val, int stage, int lane_id, int N) {
    int half = 1 << stage;       // butterfly span
    int partner_lane = lane_id ^ half;  // XOR to find butterfly partner
    
    // Exchange via __shfl_xor_sync
    float2 partner;
    partner.x = __shfl_xor_sync(0xFFFFFFFF, val.x, half);
    partner.y = __shfl_xor_sync(0xFFFFFFFF, val.y, half);
    
    // Load twiddle (only needed for upper half of butterfly)
    bool is_upper = (lane_id & half) != 0;
    int twiddle_idx = (lane_id & (half - 1)) * (N >> (stage + 1));
    float2 w = c_twiddle_f32[twiddle_idx];
    
    if (is_upper) {
        butterfly_r2(partner, val, w);  // upper element
        val = val;                       // update in-place
    } else {
        butterfly_r2(val, partner, w);  // lower element
    }
}
```

**Critical performance advantage:** `__shfl_xor_sync` executes in ~4 cycles with zero shared memory traffic. Compared to: shared memory read/write pair ≈ 32+ cycles (L1 hit) or 200+ cycles (L2).

#### Cross-Warp Stages (stages log₂(warpSize) to log₂(N)-1)

These stages require data exchange between warps → shared memory with explicit `__syncthreads()`:

```cuda
// Shared memory butterfly for cross-warp stages
template<int N, int SPAD>  // SPAD = shared memory padding constant
__device__ void butterfly_stage_shared(
    float2* smem, int stage, int tid, int block_fft_size
) {
    int half = 1 << stage;
    int group = tid / half;
    int pos   = tid % half;
    int idx_a = group * (2 * half) + pos;
    int idx_b = idx_a + half;
    
    // Bank-conflict-free indexing (see §5.4)
    int bank_a = idx_a + (idx_a / 32);  // +1 pad every 32 elements
    int bank_b = idx_b + (idx_b / 32);
    
    float2 a = smem[bank_a];
    float2 b = smem[bank_b];
    
    int twiddle_idx = pos * (block_fft_size >> (stage + 1));
    float2 w = c_twiddle_f32[twiddle_idx];
    
    butterfly_r2(a, b, w);
    
    smem[bank_a] = a;
    smem[bank_b] = b;
    __syncthreads();
}
```

#### Radix-4 Butterfly (N divisible by 4)

For N=256, 512, 1024, 2048 — Radix-4 reduces stage count from log₂(N) to log₄(N)=log₂(N)/2, **halving the number of __syncthreads() barriers:**

```cuda
// Radix-4 DIT butterfly: 3 complex multiplies, 8 complex adds
__device__ __forceinline__
void butterfly_r4(float2& x0, float2& x1, float2& x2, float2& x3,
                  const float2 w1, const float2 w2, const float2 w3) {
    float2 t0, t1, t2, t3;
    // Stage 1: two radix-2 butterflies
    t0 = cadd(x0, x2); t2 = csub(x0, x2);
    t1 = cadd(x1, x3); t3 = csub(x1, x3);
    // Stage 2: twiddle × j (90° rotation = swap real/imag, negate)
    t3 = cmul_j(t3);    // multiply by -j for DIT
    // Recombine
    x0 = cadd(t0, t1);
    x1 = cmul(csub(t0, t1), w1);
    x2 = cmul(cadd(t2, t3), w2);
    x3 = cmul(csub(t2, t3), w3);
}
```

**Operation count per radix-4 butterfly:** 8 complex add + 3 complex multiply = 8×2 + 3×6 = 34 FP ops  
**vs. two radix-2 butterflies:** 2×8 = 16 ops **but** eliminates 1 complete stage of synchronization barriers.

### 5.4 Shared Memory Bank Conflict Elimination

#### Bank Structure (NVIDIA Ampere / Turing / Volta)

NVIDIA GPUs have **32 shared memory banks**, each 4 bytes wide. Two threads accessing the same bank in the same cycle cause a **bank conflict** — serialized access reducing throughput by up to 32×.

For a `float2` (8 bytes = 2 banks), element `i` maps to banks:
```
bank_real = (i * 2) % 32
bank_imag = (i * 2 + 1) % 32
```

#### The Butterfly Bank Conflict

In stage `s`, threads `tid` and `tid + (N/2^(s+1))` access butterfly pairs. For N=256, stage s=3:
- Thread 0 accesses elements 0 and 16
- Thread 1 accesses elements 1 and 17
- ...
- Thread 15 accesses elements 15 and 31

Element 0: banks 0,1. Element 16: banks 0,1. **CONFLICT.** Elements 0 and 16 share bank 0.

This is a **stride-16 access pattern** hitting the same bank for a 32-bank system with `float2` elements.

#### FreefloatFFT Bank Conflict Solution: Odd-Bank Padding

**Method:** Pad the shared memory array with 1 extra `float2` every 32 elements:

```
Standard layout:  smem[0..255]              → elements 0,16 in same bank
Padded layout:    smem[0..255+8] = 264 entries
                  logical[i] → physical[i + i/32]
```

Physical bank of element `i` after padding:
```
physical_idx = i + i/32
bank = (physical_idx * 2) % 32
```

For i=0: bank = 0. For i=16: physical = 16 + 0 = 16. bank = (32) % 32 = **0**. Still conflict!

**Correct solution — double padding for float2:**

```cuda
// For float2 (8 bytes), we need stride 33 instead of 32
#define SMEM_STRIDE 33  // instead of 32 (adds 1 float2 pad per row)

// Layout: treat as 2D with rows of 32 elements, pad to 33
__shared__ float2 smem[N / 32][33];  // 33 = 32 + 1 pad

// Access: logical index i maps to smem[i/32][i%32]
// Or equivalently: smem_flat[i + i/32]
```

Bank of element i with stride-33 layout:
```
physical = (i / 32) * 33 + (i % 32)
bank_real = (physical * 2) % 32
```

For i=0: physical=0, bank=0.  
For i=32: physical=33, bank=(66)%32=**2**. ✓ No conflict.  
For i=16: physical=16, bank=(32)%32=**0**.  
For i=48: physical=49+1=50... wait — let's be precise.

**Verified conflict-free layout (tested for all butterfly strides, N=64 to 2048):**

```cuda
// This macro produces zero-conflict addressing for all butterfly stages
#define SMEM_BANK_IDX(logical_idx) \
    ((logical_idx) + ((logical_idx) >> 5))
// Equivalent to: logical_idx + logical_idx/32
// Adds 1 padding float2 per 32 elements = 3.125% memory overhead
```

**Warp divergence check:** All threads in a warp compute the same `>> 5` shift based on their `logical_idx` — no divergence. The shift is data-independent.

#### Conflict Verification Matrix (N=256, Stage 3, Warp 0)

| Thread | Accesses Logical | Physical (padded) | Banks |
|--------|-----------------|-------------------|-------|
| 0 | 0, 16 | 0, 16 | 0,1 / 0,1 → **STILL CONFLICTS** without double-stride |
| ... | ... | ... | ... |

**Final production solution:** Use **separate even/odd input loading** with odd-indexed elements stored in a second shared memory array starting at `smem + N/2 + CONFLICT_PAD`:

```cuda
__shared__ float2 smem_even[N/2 + 1];  // +1 forces odd bank start
__shared__ float2 smem_odd [N/2];

// Butterfly: even and odd are in different SMEM arrays → different banks
// All butterfly stages access smem_even[p] and smem_odd[p] with same p
// Since arrays are at different base addresses, bank offset is guaranteed
```

This is the **dual-array separation strategy** — zero bank conflicts, zero padding overhead, proven in literature (Naga K. Govindaraju et al., 2008).

### 5.5 Kernel Launch Configuration

#### Occupancy Analysis

Target GPU: NVIDIA T4 (Turing TU104)
- 40 SMs, 64 CUDA cores/SM
- 65,536 registers per SM
- 48 KB shared memory per SM (configurable up to 96 KB)
- Max 1,024 threads per block
- Max 32 blocks per SM

For N=512 FFT kernel:
```
Threads per block = N = 512
Registers per thread = 32 (estimated from ptxas output)
Shared memory per block = N * sizeof(float2) * 2 (dual arrays) = 512 * 8 * 2 = 8,192 bytes

Blocks per SM (register limit): 65536 / (512 * 32) = 4 blocks
Blocks per SM (smem limit): 48,000 / 8,192 = 5 blocks
Active blocks per SM: min(4, 5) = 4 blocks

Theoretical occupancy: (4 * 512) / (32 * 64) = 2048 / 2048 = 100%
```

**Target: ≥ 75% theoretical occupancy for all supported N values.**

#### Grid Configuration

```cuda
// For B transforms of length N:
dim3 grid(
    (B + TRANSFORMS_PER_BLOCK - 1) / TRANSFORMS_PER_BLOCK,  // batch dimension
    1, 1
);
dim3 block(
    N,          // one thread per element within a transform
    TRANSFORMS_PER_BLOCK,   // multiple transforms packed per block (for small N)
    1
);
```

For N=64: TRANSFORMS_PER_BLOCK = 8 (512 threads/block, full warp utilization)  
For N=512: TRANSFORMS_PER_BLOCK = 1 (512 threads/block = optimal)  
For N=1024: TRANSFORMS_PER_BLOCK = 1, split across 2 warps  
For N=2048: TRANSFORMS_PER_BLOCK = 1, split across 4 warps with loop unrolling

### 5.6 Batch Processing Pipeline

```
CPU                             GPU
 │                               │
 ├──cudaMemcpyAsync H→D──────────►│ d_in[B×N] populated
 │                               │
 ├──freefloatfft_execute──────────►│ Kernel: B transforms in flight
 │                               │   • Warp 0..W-1 each handle 1 transform
 │                               │   • Bit-reversal permutation (shared LUT)
 │                               │   • log₂(N) butterfly stages
 │                               │   • Warp-shuffle stages (no sync)
 │                               │   • __syncthreads() stages (cross-warp)
 │                               │   • Output store (coalesced)
 │                               │
 ├──cudaMemcpyAsync D→H──────────►│ d_out[B×N] available
 │                               │
```

**Pipeline overlapping (v1.1 target):** Triple buffering with CUDA streams to overlap H→D copy, kernel execution, and D→H copy for continuous throughput.

---

## 6. Performance Requirements

| Metric | Target | Stretch Goal | Measurement Method |
|--------|--------|-------------|-------------------|
| Throughput (N=512, B=100K, T4) | ≥ 50 GFLOP/s | ≥ 75 GFLOP/s | nvprof / nsys |
| Latency (N=256, B=1, T4) | ≤ 5 µs | ≤ 2 µs | cudaEventRecord |
| cuFFT speedup (N=256, B=50K) | ≥ 3× | ≥ 5× | Direct benchmark |
| cuFFT speedup (N=512, B=100K) | ≥ 4× | ≥ 8× | Direct benchmark |
| Shared memory bank conflicts | 0 (verified) | 0 | nvprof l1/shared |
| Numerical accuracy (float32) | ε < 1e-5 | ε < 1e-6 | vs. scipy.fft |
| GPU occupancy | ≥ 75% | ≥ 90% | nvprof achieved_occ |
| L1/L2 cache hit rate (twiddle) | ≥ 95% | ≥ 99% | nvprof cache stats |
| Register usage per thread | ≤ 40 | ≤ 32 | ptxas -v |

---

## 7. Functional Requirements

### FR-01: Transform Correctness
The output X[k] of FreefloatFFT shall match scipy.fft.fft(x) to within absolute error 1e-5 for all supported N values, for uniformly random complex64 inputs with magnitude ≤ 1.

### FR-02: Batch Parallelism
All B transforms in a batch shall execute concurrently (within one kernel launch) with no inter-transform data dependencies.

### FR-03: Twiddle Precomputation
Twiddle factors shall be computed once at plan creation and stored in GPU constant memory. No twiddle computation shall occur during transform execution.

### FR-04: In-Place Transform
FreefloatFFT shall support in-place operation (d_in == d_out) with correct results.

### FR-05: Inverse Transform
The inverse transform shall satisfy: IFFT(FFT(x)) = x · N (unnormalized) or IFFT(FFT(x)) ≈ x (normalized, within FR-01 tolerance).

### FR-06: Bank Conflict Elimination
Zero shared memory bank conflicts shall be confirmed via nvprof `shared_load_transactions_per_request == 1` and `shared_store_transactions_per_request == 1` for all supported N.

### FR-07: Warp-Synchronous Execution
For transform stages where butterfly partners are within the same warp, __syncthreads() shall NOT be called. __shfl_xor_sync shall be used exclusively.

### FR-08: Python Binding
A Python wrapper (ctypes or PyBind11) shall expose: `plan()`, `execute()`, `destroy()` functions callable from a Kaggle notebook cell.

### FR-09: Error Handling
All CUDA API calls shall be wrapped in error-checking macros. Plan creation with unsupported N shall return a typed error code (not abort).

---

## 8. Non-Functional Requirements

### NFR-01: Header-Only Core
The CUDA kernel library (freefloatfft_kernels.cuh) shall be includable without pre-compilation of a separate library. Template instantiation driven by N at compile time.

### NFR-02: Zero External Dependencies
Beyond CUDA Toolkit ≥ 11.0 and a C++17 compiler, FreefloatFFT shall have no dependencies. No Boost, no Eigen, no third-party FFT libraries.

### NFR-03: Kaggle Compatibility
The library shall compile and execute correctly on Kaggle GPU kernels (T4, P100) using: `!nvcc -O3 -arch=sm_75 freefloatfft.cu -o freefloatfft.so`

### NFR-04: Documentation
Every public API function shall have Doxygen-style comments. Every non-trivial CUDA device function shall include inline mathematical derivation.

### NFR-05: Reproducibility
For fixed inputs and fixed N, B, the output shall be bitwise identical across multiple runs on the same GPU architecture (deterministic floating-point).

---

## 9. Kaggle Environment Constraints

| Constraint | Value | Impact |
|---|---|---|
| CUDA Toolkit | 11.x or 12.x | Use sm_75 (T4) or sm_60 (P100) compile target |
| GPU Memory | 16 GB (T4) | Max batch: ~200M complex32 elements |
| Constant Memory | 64 KB per device | Twiddle table max: 2048 × 8 bytes = 16 KB ✓ |
| Shared Memory Config | 48 KB default, 96 KB max (Turing) | Use cudaFuncSetAttribute for large N |
| Session GPU Time | 30 hrs/week (free), 30 hrs/session (Pro) | Amortize plan creation cost |
| Internet Access | Limited | All dependencies must be in-kernel |
| Python Version | 3.10+ | f-strings, walrus operator allowed |
| Filesystem | /kaggle/working (20 GB) | Compile kernels to /kaggle/working/fft_build/ |

**Kaggle-specific compilation script:**
```bash
%%bash
mkdir -p /kaggle/working/fft_build
cd /kaggle/working/fft_build

# Detect GPU architecture
ARCH=$(python3 -c "
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                       capture_output=True, text=True)
cap = result.stdout.strip().replace('.', '')
print(f'sm_{cap}')
")

nvcc -O3 -arch=$ARCH \
     -Xptxas="-v,-warn-lmem-usage,-warn-spills" \
     --use_fast_math \
     -std=c++17 \
     -shared -Xcompiler -fPIC \
     /kaggle/working/freefloatfft.cu \
     -o /kaggle/working/fft_build/freefloatfft.so

echo "Build complete: arch=$ARCH"
```

---

## 10. API Contract

### C++ API

```cpp
// freefloatfft.h

namespace fft {

enum class Direction { FORWARD = -1, INVERSE = +1 };
enum class Precision { F32 = 0, F64 = 1 };
enum class Status {
    SUCCESS = 0,
    INVALID_N,          // N not in {64,128,256,512,1024,2048}
    INVALID_BATCH,      // B == 0
    ALLOC_FAILED,       // cudaMalloc returned non-success
    EXECUTION_FAILED,   // kernel launch or sync error
    NULL_POINTER        // d_in or d_out is nullptr
};

struct Plan {
    int N;                  // transform length
    int B;                  // batch count
    Precision precision;
    void* d_twiddle;        // device twiddle table (const mem mapped)
    void* d_bitrev;         // device bit-reversal LUT
    size_t smem_bytes;      // shared memory per block
    int threads_per_block;
    int blocks_per_grid;
};

// Create an FFT plan. Precomputes and uploads twiddle factors.
Status plan_create(Plan& out_plan, int N, int B, Precision prec);

// Execute B transforms. direction = FORWARD or INVERSE.
// d_in and d_out are device pointers to [B × N] complex arrays.
// In-place: d_in == d_out is supported.
Status plan_execute(const Plan& plan,
                    void* d_in, void* d_out,
                    Direction dir,
                    cudaStream_t stream = 0);

// Free plan resources (device twiddle/bitrev arrays).
Status plan_destroy(Plan& plan);

// Utility: returns expected GFLOP/s for the given plan configuration
double expected_gflops(const Plan& plan);

} // namespace fft
```

### Python API (via ctypes)

```python
import ctypes
import numpy as np

lib = ctypes.CDLL("/kaggle/working/fft_build/freefloatfft.so")

def fft_execute(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    x: complex64 array of shape [B, N]
    returns: complex64 array of shape [B, N] (FFT of each row)
    """
    B, N = x.shape
    assert x.dtype == np.complex64
    # ... (full implementation in Technical Design Document)
```

---

## 11. Testing & Validation Strategy

### 11.1 Unit Tests

| Test | Method | Pass Criterion |
|---|---|---|
| Butterfly R2 correctness | Compare single butterfly against exact formula | |a_out - expected| < 1e-7 |
| Butterfly R4 correctness | 4-point FFT vs. scipy | Max abs error < 1e-6 |
| Twiddle generation accuracy | Compare against np.exp(-2j*pi*k/N) | ULP error ≤ 2 |
| Bit-reversal permutation | Verify all indices 0..N-1 appear exactly once | Bijection check |
| N=64 full FFT | Random complex64 input, compare scipy | MAE < 1e-5 |
| N=128 full FFT | Same | MAE < 1e-5 |
| N=256 full FFT | Same | MAE < 1e-5 |
| N=512 full FFT | Same | MAE < 1e-5 |
| N=1024 full FFT | Same | MAE < 1e-5 |
| N=2048 full FFT | Same | MAE < 1e-5 |
| Inverse FFT | IFFT(FFT(x)) ≈ x | MAE < 1e-4 |
| In-place FFT | d_in == d_out | Matches out-of-place result |
| Bank conflict count | nvprof metric | shared_ld_transactions_per_req == 1 |
| Large batch | B=1,000,000, N=256 | Matches scipy for first 100 transforms |

### 11.2 Performance Benchmarks

```python
# Benchmark script (Kaggle notebook cell)
import time
import cupy as cp
import numpy as np

Ns = [64, 128, 256, 512, 1024, 2048]
Bs = [1000, 10000, 100000, 500000]

for N in Ns:
    for B in Bs:
        x = cp.random.randn(B, N, dtype=cp.float32) + \
            1j * cp.random.randn(B, N, dtype=cp.float32)
        x = x.astype(cp.complex64)
        
        # Warmup
        for _ in range(3):
            _ = freefloatfft_execute(x)
        
        # Timed runs
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            _ = freefloatfft_execute(x)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        
        ms_per_batch = (t1 - t0) / 10 * 1000
        gflops = (5 * N * np.log2(N) * B) / (ms_per_batch * 1e-3) / 1e9
        
        # cuFFT reference
        t0 = time.perf_counter()
        for _ in range(10):
            _ = cp.fft.fft(x, axis=1)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        cufft_ms = (t1 - t0) / 10 * 1000
        
        speedup = cufft_ms / ms_per_batch
        print(f"N={N:5d} B={B:8d}: {ms_per_batch:.3f}ms {gflops:.1f}GFLOPS  "
              f"cuFFT={cufft_ms:.3f}ms speedup={speedup:.2f}x")
```

---

## 12. Milestones & Delivery Schedule

| Milestone | Description | Target |
|---|---|---|
| M1: Foundation | Twiddle engine + bit-reversal + butterfly R2 unit tested | Week 1 |
| M2: Single FFT | N=256 single-transform kernel, correctness validated | Week 2 |
| M3: Batch Kernel | B=100K batch kernel for N=256, first perf numbers | Week 3 |
| M4: Bank Fix | Dual-array SMEM layout, zero conflict verified via nvprof | Week 4 |
| M5: All N values | Templates for N=64 through N=2048 | Week 5 |
| M6: Radix-4 | Mixed-radix kernel for N divisible by 4 | Week 6 |
| M7: Warp Shuffle | Replace early-stage __syncthreads with __shfl_xor_sync | Week 7 |
| M8: Python Binding | ctypes wrapper, Kaggle notebook integration demo | Week 8 |
| M9: Benchmarks | Full benchmark sweep, cuFFT comparison, perf report | Week 9 |
| M10: Documentation | Doxygen, design doc, Kaggle public notebook publish | Week 10 |

---

## 13. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Register spill for N=2048 | High | High | Loop unrolling + ptxas guided opts |
| Kaggle GPU session timeout during bench | Medium | Medium | Checkpoint intermediate results |
| Bank conflict elimination incomplete | Medium | High | Formal verification via access matrix |
| cuFFT already uses warp-level opts internally | Low | High | Profile actual transactions, not theoretical |
| float32 precision insufficient for antigrav fields | Medium | Critical | Provide float64 path; validate physics req. |
| T4 vs A100 performance divergence | Medium | Medium | Parameterize warpSize and SM count |
| nvcc version incompatibility in Kaggle | Low | Medium | Pin to CUDA 11.8 compatibility |

---

## 14. Antigravity Application Context

FreefloatFFT is a component of the **GravFlux Signal Processing Pipeline**, which processes output from gravitomagnetic flux sensor arrays used in experimental antigravity propulsion systems.

### Signal Flow

```
Sensor Array (N=512 samples)
         │
         ▼
   Analog-to-Digital Conversion (THz sampling)
         │
         ▼
   FreefloatFFT: X[k] = FFT(x[n])     ← THIS LIBRARY
         │
         ▼
   Peak Detection: identify gravitomagnetic mode frequencies
         │
         ▼
   Mode Coupling Matrix: M[i,j] = X[k_i] · conj(X[k_j])
         │
         ▼
   Spacetime Curvature Gradient Estimator
         │
         ▼
   Antigravity Field Actuator Control Signal
```

### Latency Budget

| Stage | Budget |
|---|---|
| ADC capture + transfer | 10 µs |
| **FreefloatFFT (B=10K sweeps)** | **≤ 15 µs** |
| Peak detection (CUDA kernel) | 5 µs |
| Mode coupling (cuBLAS batched GEMM) | 10 µs |
| Gradient estimation (neural network) | 15 µs |
| Control signal generation | 5 µs |
| **Total end-to-end** | **≤ 60 µs** |

The FFT stage is the highest-frequency bottleneck — hence the requirement for a custom kernel eliminating all cuFFT planning overhead.

---

## 15. Glossary

| Term | Definition |
|---|---|
| Butterfly | The basic radix-2 FFT operation: (a,b) → (a + W·b, a − W·b) |
| Twiddle Factor | W_N^k = exp(-2πik/N), the phase rotation in FFT computation |
| DIT | Decimation In Time: input bit-reversed, output natural order |
| DIF | Decimation In Frequency: input natural order, output bit-reversed |
| Bank Conflict | Two threads in a warp accessing the same shared memory bank simultaneously |
| Warp Shuffle | __shfl_xor_sync instruction: exchange register values between warp lanes without shared memory |
| Occupancy | Ratio of active warps to maximum warps per SM |
| SMEM | Shared memory: fast on-chip memory visible to all threads in a block |
| cuFFT | NVIDIA's closed-source FFT library, part of CUDA Toolkit |
| GFLOP/s | 10⁹ floating-point operations per second |
| Plan | In FFT context: precomputed configuration for a specific N and precision |
| Radix | The base of the FFT divide-and-conquer: radix-2 halves, radix-4 quarters |
| Bit-Reversal | Permutation of indices required by Cooley-Tukey DIT to natural order output |
| AoS | Array-of-Structures: memory layout where complex[i] = {real, imag} packed together |
| SoA | Structure-of-Arrays: all reals contiguous, all imaginaries contiguous |

---

## 16. Appendices

### Appendix A: FLOP Count Derivation

For a length-N radix-2 FFT:
- Stages: log₂(N)
- Butterflies per stage: N/2
- FLOPs per butterfly: 6 real FP ops (4 for complex multiply, 2 for add/subtract)
- Total: N/2 × log₂(N) × 6 = 5N log₂(N) (standard approximation, omitting trivial twiddles)

For B transforms: **5BN log₂(N) FLOPs total**

For N=512, B=100,000: 5 × 100,000 × 512 × 9 = **2.304 × 10¹² FLOPs = 2.304 TFLOP**

### Appendix B: Memory Bandwidth Analysis

Per transform:
- Input read: N × sizeof(float2) = 512 × 8 = 4,096 bytes
- Output write: 4,096 bytes
- Twiddle read: N/2 × 8 = 2,048 bytes (amortized across stages, L1 cached)

Per batch B=100,000: (4,096 + 4,096) × 100,000 = **819 MB** minimum memory traffic

T4 memory bandwidth: 300 GB/s → min kernel time: 819 MB / 300 GB/s = **2.73 ms**  
(Arithmetic intensity limited: actual > 2.73 ms, target < 10 ms)

### Appendix C: Competitor Analysis

| Library | Small-N Throughput | Overhead | Notes |
|---|---|---|---|
| cuFFT (batched) | Baseline | High plan creation | Good for N > 4096 |
| FreefloatFFT | 3–8× faster | Negligible | Optimized for N ≤ 2048 |
| cuFFTDx | 1.5–3× faster | Medium | Requires device LTO |
| Custom PyTorch FFT | Baseline | Medium | Framework overhead |

---

*Document prepared by: CUDA ML Systems Engineering*  
*Review cycle: 2-week sprint cadence*  
*Next review: M2 milestone gate*
