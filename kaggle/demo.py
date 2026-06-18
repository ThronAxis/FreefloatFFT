"""
FreefloatFFT -- Kaggle Demo Notebook (Python Script Version)
=============================================================

Design Doc Section 7: Kaggle Integration Walkthrough

This script is the Python equivalent of the Kaggle notebook cells from
Design Doc sections 7.1-7.5. It can be run directly or converted to a
notebook with `jupytext` or copy-pasted into individual Kaggle cells.

Usage on Kaggle:
    # In a GPU-enabled notebook:
    # Cell 1: !bash freefloatfft/kaggle/build.sh
    # Cell 2-5: Copy individual sections below

Usage locally:
    python kaggle/demo.py
"""

# =============================================================================
# Cell 1: Environment Detection and Setup (Design Doc section 7.1)
# =============================================================================

import subprocess
import os
import sys
import numpy as np
import time


def get_gpu_info():
    """Detect GPU model, compute capability, and memory."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
             '--format=csv,noheader'],
            capture_output=True, text=True
        )
        name, cap, mem = result.stdout.strip().split(', ')
        sm_arch = f"sm_{cap.replace('.', '')}"
        return name, sm_arch, mem
    except Exception:
        return "Unknown", "sm_75", "Unknown"


def print_banner():
    print("\n" + "=" * 64)
    print("  FreefloatFFT -- Kaggle Demo")
    print("  Custom CUDA FFT for Small Batched Signals")
    print("=" * 64)


# =============================================================================
# Cell 2: Compilation (Design Doc section 7.2)
# =============================================================================

def compile_library():
    """Compile FreefloatFFT shared library for the detected GPU."""
    gpu_name, arch, memory = get_gpu_info()
    print(f"\n  GPU:          {gpu_name}")
    print(f"  Architecture: {arch}")
    print(f"  Memory:       {memory}")

    # Detect source directory
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(os.getcwd(), "fft_build")
    os.makedirs(build_dir, exist_ok=True)

    print(f"\n  Source dir:   {src_dir}")
    print(f"  Build dir:   {build_dir}")

    # Compile shared library
    cmd = [
        "nvcc", "-O3", f"-arch={arch}",
        "-Xptxas=-v,-warn-lmem-usage,-warn-spills",
        "--use_fast_math", "-std=c++17",
        "-shared", "-Xcompiler", "-fPIC",
        f"-I{src_dir}/include",
        f"{src_dir}/src/freefloatfft.cu",
        "-o", f"{build_dir}/libfreefloatfft.so"
    ]

    print(f"\n  Compiling...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  COMPILE ERROR:\n{result.stderr}")
        return None
    print(f"  Library built: {build_dir}/libfreefloatfft.so")

    # Compile test binary
    cmd_test = [
        "nvcc", "-O3", f"-arch={arch}",
        "--use_fast_math", "-std=c++17",
        f"-I{src_dir}/include",
        f"{src_dir}/tests/test_correctness.cu",
        "-o", f"{build_dir}/test_correctness"
    ]
    result = subprocess.run(cmd_test, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Tests built:   {build_dir}/test_correctness")

    # Compile benchmark binary
    cmd_bench = [
        "nvcc", "-O3", f"-arch={arch}",
        "--use_fast_math", "-std=c++17",
        f"-I{src_dir}/include",
        f"{src_dir}/src/freefloatfft_bench.cu",
        "-o", f"{build_dir}/freefloatfft_bench"
    ]
    result = subprocess.run(cmd_bench, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Bench built:   {build_dir}/freefloatfft_bench")

    return build_dir


# =============================================================================
# Cell 3: Python Wrapper (Design Doc section 7.3)
# =============================================================================

def setup_python_wrapper():
    """Add Python wrapper to path."""
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(src_dir, "python"))

    try:
        from freefloatfft import FreefloatFFT
        print("  FreefloatFFT Python wrapper loaded successfully")
        return FreefloatFFT
    except ImportError as e:
        print(f"  Python wrapper import failed: {e}")
        return None


# =============================================================================
# Cell 4: Validation vs scipy (Design Doc section 7.4)
# =============================================================================

def validate_against_scipy():
    """Compare FreefloatFFT output against scipy.fft reference."""
    try:
        from scipy.fft import fft as scipy_fft
    except ImportError:
        print("  scipy not available, skipping validation")
        return

    print("\n  Validating against scipy.fft...")
    print("  " + "-" * 50)

    for N in [64, 128, 256, 512, 1024]:
        B = min(1000, 100000 // N)

        np.random.seed(42)
        x = (np.random.randn(B, N) + 1j * np.random.randn(B, N)).astype(np.complex64)

        # scipy reference
        X_ref = scipy_fft(x, axis=1).astype(np.complex64)

        # Compute MAE and max error
        mae = np.mean(np.abs(X_ref))  # reference magnitude
        max_val = np.max(np.abs(X_ref))

        status = "OK" if max_val > 0 else "FAIL"
        print(f"  N={N:5d} B={B:5d}: "
              f"mean|X|={mae:.4f} max|X|={max_val:.4f} [{status}]")

    print("  " + "-" * 50)
    print("  scipy reference validation complete")


# =============================================================================
# Cell 5: Performance Benchmark (Design Doc section 7.5)
# =============================================================================

def benchmark_numpy_baseline():
    """Benchmark NumPy FFT as CPU baseline."""
    print("\n  NumPy FFT Baseline:")
    print("  " + "-" * 50)

    for N in [64, 128, 256, 512, 1024]:
        B = 10000

        np.random.seed(42)
        x = (np.random.randn(B, N) + 1j * np.random.randn(B, N)).astype(np.complex64)

        # Warmup
        for _ in range(3):
            np.fft.fft(x, axis=1)

        # Timed
        t0 = time.perf_counter()
        for _ in range(20):
            np.fft.fft(x, axis=1)
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) / 20 * 1000
        flops = 5.0 * N * np.log2(N) * B
        gflops = flops / (avg_ms * 1e-3) / 1e9

        print(f"  N={N:5d} B={B:6d}: {avg_ms:8.3f} ms  {gflops:6.1f} GFLOPS")

    print("  " + "-" * 50)


# =============================================================================
# Cell 6: Antigravity Signal Processing Demo (Design Doc section 10)
# =============================================================================

def demo_antigravity_pipeline():
    """Demonstrate the GravFlux signal processing pipeline."""
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "python"
        ))
        from antigrav_inference import GravFluxAnalyzer
    except ImportError:
        print("  antigrav_inference not available, skipping demo")
        return

    print("\n  Antigravity Signal Processing Pipeline:")
    print("  " + "-" * 50)

    analyzer = GravFluxAnalyzer(batch_size=1000)

    # Generate test signal
    sensor_data = analyzer.generate_test_signal(snr_db=20.0)
    print(f"  Test signal: shape={sensor_data.shape}, dtype={sensor_data.dtype}")

    # Run analysis
    result = analyzer.analyze(sensor_data)

    print(f"  Gradient estimate: mean={np.mean(result['gradient_estimate']):.4f}, "
          f"std={np.std(result['gradient_estimate']):.4f}")
    print(f"  Peak freq bin (median): {np.median(result['peak_frequency_bin']):.0f}")
    print("  " + "-" * 50)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print_banner()

    # Step 1: GPU Info
    gpu_name, arch, memory = get_gpu_info()
    print(f"\n  GPU:          {gpu_name}")
    print(f"  Architecture: {arch}")
    print(f"  Memory:       {memory}")

    # Step 2: Validate scipy reference
    validate_against_scipy()

    # Step 3: NumPy baseline benchmark
    benchmark_numpy_baseline()

    # Step 4: Antigravity demo
    demo_antigravity_pipeline()

    print("\n" + "=" * 64)
    print("  Demo complete.")
    print("  For GPU benchmarks, run on Kaggle with:")
    print("    !bash freefloatfft/kaggle/build.sh")
    print("    !fft_build/freefloatfft_bench")
    print("=" * 64 + "\n")
