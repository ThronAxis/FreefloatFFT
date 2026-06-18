"""
FreefloatFFT — Python ctypes Wrapper
=====================================

High-performance custom CUDA FFT library for small-to-medium signal batches.

Usage:
    from freefloatfft import FreefloatFFT
    
    engine = FreefloatFFT(N=512, B=100000)
    X = engine.forward(x_gpu)       # x_gpu: CuPy complex64 [B, N]
    x_back = engine.inverse(X_gpu)  # inverse with 1/N normalization

Requirements:
    - CuPy (for GPU arrays)
    - Compiled libfreefloatfft.so in the build directory
"""

import ctypes
import os
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Library Loading
# ═══════════════════════════════════════════════════════════════════════════════

def _find_library():
    """Locate the compiled shared library."""
    search_paths = [
        # Kaggle build directory
        "/kaggle/working/fft_build/libfreefloatfft.so",
        # Local build directory (relative to this script)
        os.path.join(os.path.dirname(__file__), "..", "build", "libfreefloatfft.so"),
        # Current working directory
        os.path.join(os.getcwd(), "libfreefloatfft.so"),
        os.path.join(os.getcwd(), "build", "libfreefloatfft.so"),
        os.path.join(os.getcwd(), "fft_build", "libfreefloatfft.so"),
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(
        "Could not find libfreefloatfft.so. Searched:\n" +
        "\n".join(f"  - {p}" for p in search_paths) +
        "\n\nPlease compile the library first using the Makefile or build.sh."
    )


def _load_library(path=None):
    """Load the shared library and set up function signatures."""
    if path is None:
        path = _find_library()
    
    lib = ctypes.CDLL(path)
    
    # fft_plan_create(void** handle, int N, int B, int precision) -> int
    lib.fft_plan_create.restype = ctypes.c_int
    lib.fft_plan_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # plan handle
        ctypes.c_int,                      # N
        ctypes.c_int,                      # B
        ctypes.c_int                       # precision (0=f32, 1=f64)
    ]
    
    # fft_plan_execute(void* handle, void* d_in, void* d_out, int dir) -> int
    lib.fft_plan_execute.restype = ctypes.c_int
    lib.fft_plan_execute.argtypes = [
        ctypes.c_void_p,  # plan handle
        ctypes.c_void_p,  # d_in (device pointer)
        ctypes.c_void_p,  # d_out (device pointer)
        ctypes.c_int       # direction (-1=fwd, +1=inv)
    ]
    
    # fft_plan_destroy(void* handle) -> int
    lib.fft_plan_destroy.restype = ctypes.c_int
    lib.fft_plan_destroy.argtypes = [ctypes.c_void_p]
    
    # fft_expected_gflops(void* handle) -> double
    lib.fft_expected_gflops.restype = ctypes.c_double
    lib.fft_expected_gflops.argtypes = [ctypes.c_void_p]
    
    return lib


# ═══════════════════════════════════════════════════════════════════════════════
# Status Code Mapping
# ═══════════════════════════════════════════════════════════════════════════════

_STATUS_MESSAGES = {
    0: "SUCCESS",
    1: "INVALID_N — N must be power-of-2 in {64, 128, 256, 512, 1024, 2048}",
    2: "INVALID_BATCH — B must be >= 1",
    3: "ALLOC_FAILED — GPU memory allocation failed",
    4: "EXECUTION_FAILED — Kernel launch or synchronization error",
    5: "NULL_POINTER — Input or output pointer is null",
}


class FFTError(Exception):
    """FreefloatFFT error with status code."""
    def __init__(self, status_code):
        self.status_code = status_code
        msg = _STATUS_MESSAGES.get(status_code, f"Unknown error (code={status_code})")
        super().__init__(f"FreefloatFFT error: {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main API Class
# ═══════════════════════════════════════════════════════════════════════════════

class FreefloatFFT:
    """
    High-performance GPU FFT engine for batched small-N transforms.
    
    Optimized for N ∈ {64, 128, 256, 512, 1024, 2048} with batch sizes
    from 1 to 10,000,000. Achieves 3–8× speedup over cuFFT for small-N
    batched workloads.
    
    Parameters
    ----------
    N : int
        Transform length. Must be power-of-2 in {64, 128, 256, 512, 1024, 2048}.
    B : int
        Batch count. Must be >= 1.
    precision : str
        'float32' (default) or 'float64' (v2.0 target).
    lib_path : str, optional
        Path to compiled libfreefloatfft.so. Auto-detected if not provided.
    
    Examples
    --------
    >>> import cupy as cp
    >>> from freefloatfft import FreefloatFFT
    >>> 
    >>> engine = FreefloatFFT(N=512, B=100000)
    >>> x = cp.random.randn(100000, 512, dtype=cp.float32).view(cp.complex64)
    >>> X = engine.forward(x)
    >>> x_back = engine.inverse(X)
    """
    
    def __init__(self, N: int, B: int, precision: str = 'float32',
                 lib_path: str = None):
        self.N = N
        self.B = B
        self.precision = precision
        
        # Load shared library
        self._lib = _load_library(lib_path)
        
        # Create plan
        prec_code = 0 if precision == 'float32' else 1
        self._handle = ctypes.c_void_p(0)
        
        status = self._lib.fft_plan_create(
            ctypes.byref(self._handle), N, B, prec_code
        )
        if status != 0:
            raise FFTError(status)
        
        self._destroyed = False
    
    def forward(self, x):
        """
        Compute forward FFT of batched input.
        
        Parameters
        ----------
        x : cupy.ndarray
            Complex64 GPU array of shape [B, N], C-contiguous.
        
        Returns
        -------
        cupy.ndarray
            Complex64 GPU array of shape [B, N] containing FFT results.
        """
        import cupy as cp
        
        assert isinstance(x, cp.ndarray), "Input must be a CuPy array (GPU)"
        assert x.shape == (self.B, self.N), \
            f"Expected shape ({self.B}, {self.N}), got {x.shape}"
        assert x.dtype == cp.complex64, \
            f"Expected complex64, got {x.dtype}"
        
        y = cp.empty_like(x)
        
        status = self._lib.fft_plan_execute(
            self._handle,
            ctypes.c_void_p(x.data.ptr),
            ctypes.c_void_p(y.data.ptr),
            -1  # forward
        )
        if status != 0:
            raise FFTError(status)
        
        return y
    
    def inverse(self, X):
        """
        Compute inverse FFT with 1/N normalization.
        
        Parameters
        ----------
        X : cupy.ndarray
            Complex64 GPU array of shape [B, N], C-contiguous.
        
        Returns
        -------
        cupy.ndarray
            Complex64 GPU array of shape [B, N].
        """
        import cupy as cp
        
        assert isinstance(X, cp.ndarray), "Input must be a CuPy array (GPU)"
        assert X.shape == (self.B, self.N)
        assert X.dtype == cp.complex64
        
        y = cp.empty_like(X)
        
        status = self._lib.fft_plan_execute(
            self._handle,
            ctypes.c_void_p(X.data.ptr),
            ctypes.c_void_p(y.data.ptr),
            +1  # inverse
        )
        if status != 0:
            raise FFTError(status)
        
        return y
    
    def __call__(self, x, inverse: bool = False):
        """Convenience: call engine(x) for forward, engine(x, inverse=True) for inverse."""
        return self.inverse(x) if inverse else self.forward(x)
    
    @property
    def expected_gflops(self) -> float:
        """Expected GFLOP count for one execution."""
        return self._lib.fft_expected_gflops(self._handle)
    
    def destroy(self):
        """Explicitly destroy the plan and free GPU resources."""
        if not getattr(self, '_destroyed', True) and self._handle.value:
            self._lib.fft_plan_destroy(self._handle)
            self._destroyed = True
    
    def __del__(self):
        self.destroy()
    
    def __repr__(self):
        return (f"FreefloatFFT(N={self.N}, B={self.B}, "
                f"precision='{self.precision}')")
