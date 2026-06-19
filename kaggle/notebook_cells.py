"""
FreefloatFFT -- Kaggle GPU Notebook Cells
==========================================

Copy each section into a separate Kaggle notebook cell.
Requires: GPU-enabled Kaggle notebook (T4/P100/A100).

Upload the entire freefloatfft/ directory to /kaggle/working/ first.
"""

# ============================================================
# CELL 1: Build Library + Tests + Benchmarks
# ============================================================
# %%bash
# cd /kaggle/working/freefloatfft
# bash kaggle/build.sh

# ============================================================
# CELL 2: Run Correctness Tests (CUDA)
# ============================================================
# !!/kaggle/working/fft_build/test_correctness

# ============================================================
# CELL 3: Run Benchmarks
# ============================================================
# !!/kaggle/working/fft_build/freefloatfft_bench

# ============================================================
# CELL 4: Python Validation vs scipy
# ============================================================

import sys, os
sys.path.insert(0, '/kaggle/working/freefloatfft/python')
os.environ['LD_LIBRARY_PATH'] = '/kaggle/working/fft_build'

import numpy as np
from scipy.fft import fft as scipy_fft

print("=" * 60)
print("  FreefloatFFT vs scipy.fft Validation")
print("=" * 60)

for N in [64, 128, 256, 512, 1024]:
    B = 100
    np.random.seed(42 + N)
    x = (np.random.randn(B, N) + 1j * np.random.randn(B, N)).astype(np.complex64)
    X_ref = scipy_fft(x, axis=1).astype(np.complex64)

    # Spot-check with naive DFT
    X_naive = np.zeros((min(B,5), N), dtype=np.complex64)
    for b in range(min(B, 5)):
        for k in range(N):
            for n in range(N):
                X_naive[b, k] += x[b, n] * np.exp(-2j * np.pi * k * n / N)

    err = np.max(np.abs(X_ref[:5] - X_naive))
    print(f"  N={N:5d}: max_err={err:.2e} [{'PASS' if err < 1e-2 else 'FAIL'}]")

# ============================================================
# CELL 5: GPU Benchmark (CuPy vs FreefloatFFT)
# ============================================================

try:
    import cupy as cp
    from freefloatfft import FreefloatFFT
    import time

    print("\n" + "=" * 60)
    print("  GPU Benchmark: FreefloatFFT vs cuFFT (CuPy)")
    print("=" * 60)

    for N in [64, 128, 256, 512, 1024]:
        B = 100000
        engine = FreefloatFFT(N=N, B=B)
        x_gpu = (cp.random.randn(B, N) + 1j * cp.random.randn(B, N)).astype(cp.complex64)

        # Warmup
        for _ in range(5): engine.forward(x_gpu)
        for _ in range(5): cp.fft.fft(x_gpu, axis=1)
        cp.cuda.Stream.null.synchronize()

        # FreefloatFFT
        t0 = time.perf_counter()
        for _ in range(20): engine.forward(x_gpu)
        cp.cuda.Stream.null.synchronize()
        ff_ms = (time.perf_counter() - t0) / 20 * 1000

        # cuFFT
        t0 = time.perf_counter()
        for _ in range(20): cp.fft.fft(x_gpu, axis=1)
        cp.cuda.Stream.null.synchronize()
        cu_ms = (time.perf_counter() - t0) / 20 * 1000

        speedup = cu_ms / ff_ms if ff_ms > 0 else 0
        gflops = 5 * N * np.log2(N) * B / (ff_ms * 1e-3) / 1e9
        mark = "[OK]" if speedup >= 3.0 else "[!]"
        print(f"  N={N:5d} B={B:6d}: FF={ff_ms:.3f}ms cuFFT={cu_ms:.3f}ms "
              f"speedup={speedup:.2f}x {gflops:.0f}GFLOPS {mark}")
        engine.destroy()

except ImportError as e:
    print(f"  GPU benchmark skipped: {e}")

# ============================================================
# CELL 6: Antigravity Signal Processing Demo
# ============================================================

from antigrav_inference import GravFluxAnalyzer

analyzer = GravFluxAnalyzer(batch_size=1000)
data = analyzer.generate_test_signal(snr_db=20.0)
result = analyzer.analyze(data)

print(f"\n  Gradient: mean={np.mean(result['gradient_estimate']):.4f}")
print(f"  Peak bin: {np.median(result['peak_frequency_bin']):.0f}")
analyzer.latency_budget()
