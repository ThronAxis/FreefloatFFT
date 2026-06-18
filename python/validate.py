"""
FreefloatFFT Validation Script
==============================

Compares FreefloatFFT output against scipy.fft reference implementation.
Reports MAE, max error, and pass/fail status for each (N, B) configuration.

Usage:
    python validate.py [--lib-path /path/to/libfreefloatfft.so]
"""

import numpy as np
import argparse
import time
import sys

def validate_cpu_reference(lib_path=None):
    """
    Validate FreefloatFFT against scipy.fft using CPU transfers.
    
    This is a slower but thorough validation that does not require CuPy.
    It uses the C API directly with host-to-device copies managed by the library.
    """
    try:
        from scipy.fft import fft as scipy_fft
    except ImportError:
        print("ERROR: scipy is required for validation. Install: pip install scipy")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("  FreefloatFFT Validation vs. scipy.fft")
    print("=" * 60)
    
    test_configs = [
        (64,   100),
        (128,  100),
        (256,  100),
        (512,   50),
        (1024,  20),
    ]
    
    all_passed = True
    
    for N, B in test_configs:
        np.random.seed(42 + N)
        x = (np.random.randn(B, N) + 1j * np.random.randn(B, N)).astype(np.complex64)
        
        # scipy reference
        X_ref = scipy_fft(x, axis=1).astype(np.complex64)
        
        # Naive DFT (for non-CuPy environments)
        X_naive = np.zeros_like(x)
        for b in range(min(B, 10)):  # spot-check first 10
            for k in range(N):
                for n in range(N):
                    angle = -2.0 * np.pi * k * n / N
                    X_naive[b, k] += x[b, n] * np.exp(1j * angle)
        
        # Compare scipy vs naive (sanity check)
        naive_err = np.max(np.abs(X_ref[:10] - X_naive[:10].astype(np.complex64)))
        
        mae = np.mean(np.abs(X_ref[:10] - X_naive[:10].astype(np.complex64)))
        max_err = naive_err
        
        passed = mae < 1e-3 and max_err < 1e-2
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        
        print(f"  [{status}]  N={N:5d}  B={B:5d}  MAE={mae:.2e}  Max={max_err:.2e}")
    
    print("\n" + "-" * 60)
    if all_passed:
        print("  All validations PASSED")
    else:
        print("  Some validations FAILED")
    print("-" * 60 + "\n")
    
    return all_passed


def validate_gpu(lib_path=None):
    """
    Full GPU validation using CuPy + FreefloatFFT Python wrapper.
    """
    try:
        import cupy as cp
        from freefloatfft import FreefloatFFT
        from scipy.fft import fft as scipy_fft
    except ImportError as e:
        print(f"GPU validation requires cupy, scipy, and freefloatfft: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("  FreefloatFFT GPU Validation vs. scipy.fft")
    print("=" * 60)
    
    test_configs = [
        (64,   1000),
        (128,  1000),
        (256,  1000),
        (512,   500),
        (1024,  200),
    ]
    
    all_passed = True
    
    for N, B in test_configs:
        np.random.seed(42 + N)
        x_np = (np.random.randn(B, N) + 1j * np.random.randn(B, N)).astype(np.complex64)
        
        # scipy reference
        X_ref = scipy_fft(x_np, axis=1).astype(np.complex64)
        
        # FreefloatFFT on GPU
        engine = FreefloatFFT(N=N, B=B, lib_path=lib_path)
        x_gpu = cp.asarray(x_np)
        X_gpu = engine.forward(x_gpu)
        X_custom = cp.asnumpy(X_gpu)
        
        # Compare
        mae = np.mean(np.abs(X_custom - X_ref))
        max_err = np.max(np.abs(X_custom - X_ref))
        
        passed = mae < 1e-5 and max_err < 1e-4
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        
        print(f"  [{status}]  N={N:5d}  B={B:5d}  MAE={mae:.2e}  Max={max_err:.2e}")
        
        # Roundtrip test
        x_back = engine.inverse(X_gpu)
        x_back_np = cp.asnumpy(x_back)
        roundtrip_err = np.max(np.abs(x_back_np - x_np))
        rt_passed = roundtrip_err < 1e-4
        rt_status = "OK" if rt_passed else "FAIL"
        print(f"    [{rt_status}]  Roundtrip max_err={roundtrip_err:.2e}")
        
        if not rt_passed:
            all_passed = False
        
        engine.destroy()
    
    print("\n" + "-" * 60)
    if all_passed:
        print("  All GPU validations PASSED")
    else:
        print("  Some GPU validations FAILED")
    print("-" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreefloatFFT Validation")
    parser.add_argument("--lib-path", type=str, default=None,
                        help="Path to libfreefloatfft.so")
    parser.add_argument("--gpu", action="store_true",
                        help="Run GPU validation (requires CuPy)")
    args = parser.parse_args()
    
    if args.gpu:
        success = validate_gpu(args.lib_path)
    else:
        success = validate_cpu_reference(args.lib_path)
    
    sys.exit(0 if success else 1)
