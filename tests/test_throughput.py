"""
FreefloatFFT -- Performance Throughput Benchmarks
==================================================

Design Doc Section 9: Benchmark Results & Analysis

Benchmarks FreefloatFFT against cuFFT (via CuPy) across all supported
N values and batch sizes. Generates a results table matching the expected
profile from Design Doc section 9.1.

Expected Results Profile (T4, CUDA 11.8):
    N=64,  B=100K:  0.8ms  -> 4.0x speedup, 48 GFLOPS
    N=128, B=100K:  1.4ms  -> 3.6x speedup, 72 GFLOPS
    N=256, B=100K:  2.1ms  -> 3.7x speedup, 98 GFLOPS
    N=512, B=100K:  3.8ms  -> 4.8x speedup, 87 GFLOPS
    N=1024,B=50K:   3.5ms  -> 6.1x speedup, 75 GFLOPS

Usage:
    python test_throughput.py                    # full sweep
    python test_throughput.py --N 512 --B 100000 # single config
"""

import argparse
import sys
import time
import numpy as np


def benchmark_numpy_fft(N, B, iters=20):
    """Benchmark numpy.fft as CPU baseline."""
    np.random.seed(42)
    x = (np.random.randn(B, N) + 1j * np.random.randn(B, N)).astype(np.complex64)

    # Warmup
    for _ in range(3):
        np.fft.fft(x, axis=1)

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(iters):
        np.fft.fft(x, axis=1)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / iters * 1000
    flops = 5.0 * N * np.log2(N) * B
    gflops = flops / (avg_ms * 1e-3) / 1e9

    return avg_ms, gflops


def benchmark_cufft(N, B, iters=20):
    """Benchmark cuFFT via CuPy."""
    try:
        import cupy as cp
    except ImportError:
        return None, None

    x_gpu = (cp.random.randn(B, N, dtype=cp.float32) + \
             1j * cp.random.randn(B, N, dtype=cp.float32)).astype(cp.complex64)

    # Warmup
    for _ in range(5):
        cp.fft.fft(x_gpu, axis=1)
    cp.cuda.Stream.null.synchronize()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(iters):
        cp.fft.fft(x_gpu, axis=1)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / iters * 1000
    flops = 5.0 * N * np.log2(N) * B
    gflops = flops / (avg_ms * 1e-3) / 1e9

    return avg_ms, gflops


def benchmark_freefloatfft(N, B, iters=20, lib_path=None):
    """Benchmark FreefloatFFT via Python wrapper."""
    try:
        import cupy as cp
        sys.path.insert(0, '.')
        from freefloatfft import FreefloatFFT

        engine = FreefloatFFT(N=N, B=B, lib_path=lib_path)

        x_gpu = (cp.random.randn(B, N, dtype=cp.float32) + \
                 1j * cp.random.randn(B, N, dtype=cp.float32)).astype(cp.complex64)

        # Warmup
        for _ in range(5):
            engine.forward(x_gpu)
        cp.cuda.Stream.null.synchronize()

        # Timed runs
        t0 = time.perf_counter()
        for _ in range(iters):
            engine.forward(x_gpu)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) / iters * 1000
        flops = 5.0 * N * np.log2(N) * B
        gflops = flops / (avg_ms * 1e-3) / 1e9

        engine.destroy()
        return avg_ms, gflops
    except Exception as e:
        print(f"  FreefloatFFT benchmark failed: {e}")
        return None, None


def run_full_sweep(lib_path=None, iters=20):
    """Run benchmark sweep across all N and B configurations."""
    print("\n" + "=" * 80)
    print("  FreefloatFFT Performance Benchmark Suite")
    print("  Design Doc Section 9: Benchmark Results & Analysis")
    print("=" * 80)

    # Check GPU availability
    has_gpu = False
    try:
        import cupy as cp
        prop = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = prop['name'].decode() if isinstance(prop['name'], bytes) else prop['name']
        print(f"\n  GPU: {gpu_name}")
        has_gpu = True
    except Exception:
        print("\n  GPU: Not available (CPU-only benchmark)")

    Ns = [64, 128, 256, 512, 1024]
    Bs = [1_000, 10_000, 100_000]

    print(f"\n  {'N':>6} {'B':>10} | {'numpy (ms)':>12} {'cuFFT (ms)':>12} "
          f"{'FFT (ms)':>12} | {'Speedup':>8} {'GFLOPS':>8}")
    print("  " + "-" * 76)

    results = []

    for N in Ns:
        for B in Bs:
            # NumPy baseline
            np_ms, np_gf = benchmark_numpy_fft(N, B, iters)

            # cuFFT via CuPy
            cu_ms, cu_gf = (None, None)
            if has_gpu:
                cu_ms, cu_gf = benchmark_cufft(N, B, iters)

            # FreefloatFFT
            ff_ms, ff_gf = (None, None)
            if has_gpu:
                ff_ms, ff_gf = benchmark_freefloatfft(N, B, iters, lib_path)

            # Compute speedup
            speedup = "-"
            if cu_ms and ff_ms and ff_ms > 0:
                speedup = f"{cu_ms / ff_ms:.2f}x"

            cu_str = f"{cu_ms:.3f}" if cu_ms else "N/A"
            ff_str = f"{ff_ms:.3f}" if ff_ms else "N/A"
            gf_str = f"{ff_gf:.1f}" if ff_gf else "-"

            marker = ""
            if cu_ms and ff_ms and ff_ms > 0:
                ratio = cu_ms / ff_ms
                if ratio >= 3.0:
                    marker = " [OK]"
                else:
                    marker = " [!]"

            print(f"  {N:>6} {B:>10,} | {np_ms:>12.3f} {cu_str:>12} "
                  f"{ff_str:>12} | {speedup:>8} {gf_str:>8}{marker}")

            results.append({
                'N': N, 'B': B,
                'numpy_ms': round(np_ms, 4),
                'cufft_ms': round(cu_ms, 4) if cu_ms else None,
                'freefloatfft_ms': round(ff_ms, 4) if ff_ms else None,
                'speedup': speedup,
                'gflops': round(ff_gf, 1) if ff_gf else None
            })

    print("\n  " + "-" * 76)
    print("  [OK] = >= 3x speedup over cuFFT (PRD target)")
    print("  [!]  = Below target speedup")
    print("=" * 80 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="FreefloatFFT Throughput Benchmarks")
    parser.add_argument("--N", type=int, default=0, help="FFT length (0 = full sweep)")
    parser.add_argument("--B", type=int, default=0, help="Batch size (0 = full sweep)")
    parser.add_argument("--iters", type=int, default=20, help="Iterations per measurement")
    parser.add_argument("--lib-path", type=str, default=None, help="Path to libfreefloatfft.so")
    args = parser.parse_args()

    if args.N > 0 and args.B > 0:
        print(f"\n  Single benchmark: N={args.N}, B={args.B}")

        np_ms, np_gf = benchmark_numpy_fft(args.N, args.B, args.iters)
        print(f"  NumPy FFT:       {np_ms:.3f} ms  ({np_gf:.1f} GFLOPS)")

        cu_ms, cu_gf = benchmark_cufft(args.N, args.B, args.iters)
        if cu_ms:
            print(f"  cuFFT (CuPy):    {cu_ms:.3f} ms  ({cu_gf:.1f} GFLOPS)")

        ff_ms, ff_gf = benchmark_freefloatfft(args.N, args.B, args.iters, args.lib_path)
        if ff_ms:
            print(f"  FreefloatFFT:    {ff_ms:.3f} ms  ({ff_gf:.1f} GFLOPS)")
            if cu_ms:
                print(f"  Speedup:         {cu_ms/ff_ms:.2f}x over cuFFT")
    else:
        run_full_sweep(args.lib_path, args.iters)


if __name__ == "__main__":
    main()
