/**
 * @file freefloatfft.cu
 * @brief FreefloatFFT -- Plan management, twiddle upload, and C API exports
 *
 * This is the main compilation unit. It includes all header-only kernel
 * templates and provides the plan_create/execute/destroy API as well as
 * C-linkage exports for Python ctypes integration.
 *
 * Inverse FFT correctness:
 *   The kernel stages always read from c_twiddle_f32 (the active twiddle slot).
 *   Before each execute call, we check if the direction has changed and swap
 *   the active twiddle table (forward vs inverse) accordingly. This ensures
 *   IFFT(FFT(x)) = x with correct conjugate twiddle factors.
 *
 * Compilation:
 *   nvcc -O3 -arch=sm_75 --use_fast_math -std=c++17 \
 *        -shared -Xcompiler -fPIC freefloatfft.cu -o libfreefloatfft.so
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// Include all kernel headers (header-only library)
#include "../include/freefloatfft.h"
#include "../include/freefloatfft_math.cuh"
#include "../include/freefloatfft_constants.cuh"
#include "../include/freefloatfft_warp.cuh"
#include "../include/freefloatfft_smem.cuh"
#include "../include/freefloatfft_kernel.cuh"

// =============================================================================
// Plan Create
// =============================================================================

namespace fft {

Status plan_create(Plan& plan, int N, int B, Precision prec) {
    // -- Validation -----------------------------------------------------------
    if (!is_valid_N(N)) {
        fprintf(stderr, "FreefloatFFT: Invalid N=%d. Must be power-of-2 in [64, 2048]\n", N);
        return Status::INVALID_N;
    }
    if (B <= 0) {
        fprintf(stderr, "FreefloatFFT: Invalid B=%d. Must be >= 1\n", B);
        return Status::INVALID_BATCH;
    }

    // -- Initialize plan fields -----------------------------------------------
    memset(&plan, 0, sizeof(Plan));
    plan.N = N;
    plan.B = B;
    plan.log2N = log2_N(N);
    plan.precision = prec;
    plan.initialized = false;
    plan.last_direction = 0;   // no direction set yet

    // -- Kernel launch configuration ------------------------------------------
    plan.threads_per_block = (N <= 1024) ? N : 1024;
    plan.transforms_per_block = 1;
    plan.blocks_per_grid = B;

    // Shared memory: dual-array layout = (N + 2) x sizeof(float2)
    plan.smem_bytes = (N + 2) * sizeof(float2);

    // -- Precompute twiddle factors (double precision) -------------------------
    int half_N = N / 2;

    // Allocate host buffers (persist for direction swapping in plan_execute)
    plan.h_twiddle_fwd = (float2*)malloc(half_N * sizeof(float2));
    plan.h_twiddle_inv = (float2*)malloc(half_N * sizeof(float2));
    uint16_t* h_bitrev = (uint16_t*)malloc(N * sizeof(uint16_t));

    if (!plan.h_twiddle_fwd || !plan.h_twiddle_inv || !h_bitrev) {
        fprintf(stderr, "FreefloatFFT: host malloc failed\n");
        free(plan.h_twiddle_fwd);
        free(plan.h_twiddle_inv);
        free(h_bitrev);
        plan.h_twiddle_fwd = nullptr;
        plan.h_twiddle_inv = nullptr;
        return Status::ALLOC_FAILED;
    }

    // Compute forward twiddles: W_N = exp(-2*pi*i/N)
    precompute_twiddles_f32(plan.h_twiddle_fwd, N, -1);

    // Compute inverse twiddles: W_N = exp(+2*pi*i/N)
    precompute_twiddles_f32(plan.h_twiddle_inv, N, +1);

    // Build bit-reversal LUT
    build_bitrev_lut(h_bitrev, N);

    // Upload forward twiddles to constant memory (default active slot)
    cudaError_t err;
    err = upload_twiddles_forward(plan.h_twiddle_fwd, half_N);
    if (err != cudaSuccess) {
        fprintf(stderr, "FreefloatFFT: twiddle upload failed: %s\n",
                cudaGetErrorString(err));
        free(plan.h_twiddle_fwd);
        free(plan.h_twiddle_inv);
        free(h_bitrev);
        plan.h_twiddle_fwd = nullptr;
        plan.h_twiddle_inv = nullptr;
        return Status::ALLOC_FAILED;
    }

    // Also upload inverse twiddles to their separate constant slot
    err = upload_twiddles_inverse(plan.h_twiddle_inv, half_N);
    if (err != cudaSuccess) {
        fprintf(stderr, "FreefloatFFT: inverse twiddle upload failed: %s\n",
                cudaGetErrorString(err));
        free(plan.h_twiddle_fwd);
        free(plan.h_twiddle_inv);
        free(h_bitrev);
        plan.h_twiddle_fwd = nullptr;
        plan.h_twiddle_inv = nullptr;
        return Status::ALLOC_FAILED;
    }

    err = upload_bitrev(h_bitrev, N);
    if (err != cudaSuccess) {
        fprintf(stderr, "FreefloatFFT: bitrev upload failed: %s\n",
                cudaGetErrorString(err));
        free(plan.h_twiddle_fwd);
        free(plan.h_twiddle_inv);
        free(h_bitrev);
        plan.h_twiddle_fwd = nullptr;
        plan.h_twiddle_inv = nullptr;
        return Status::ALLOC_FAILED;
    }

    // Free bitrev host buffer (twiddle buffers are persisted in the plan)
    free(h_bitrev);

    // Mark the active twiddle slot as forward (since we just uploaded forward)
    plan.last_direction = -1;
    plan.initialized = true;

    printf("FreefloatFFT: Plan created -- N=%d, B=%d, log2N=%d, "
           "smem=%zu bytes, threads=%d, blocks=%d\n",
           plan.N, plan.B, plan.log2N,
           plan.smem_bytes, plan.threads_per_block, plan.blocks_per_grid);

    return Status::SUCCESS;
}

// =============================================================================
// Plan Execute
// =============================================================================

Status plan_execute(const Plan& plan,
                    void* d_in, void* d_out,
                    Direction dir,
                    cudaStream_t stream) {
    // -- Validation -----------------------------------------------------------
    if (!plan.initialized) {
        fprintf(stderr, "FreefloatFFT: Plan not initialized\n");
        return Status::EXECUTION_FAILED;
    }
    if (!d_in) {
        fprintf(stderr, "FreefloatFFT: d_in is null\n");
        return Status::NULL_POINTER;
    }
    if (!d_out) {
        fprintf(stderr, "FreefloatFFT: d_out is null\n");
        return Status::NULL_POINTER;
    }

    // -- Swap active twiddle table if direction changed ------------------------
    // The kernel stages always read from c_twiddle_f32.
    // If the direction differs from the last call, we re-upload the correct
    // twiddle table to the active slot. Cost: ~0.5 us for N<=1024.
    int direction = static_cast<int>(dir);
    if (direction != plan.last_direction) {
        cudaError_t err = activate_twiddles(
            direction, plan.N / 2,
            plan.h_twiddle_fwd, plan.h_twiddle_inv
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "FreefloatFFT: twiddle swap failed: %s\n",
                    cudaGetErrorString(err));
            return Status::EXECUTION_FAILED;
        }
        // Safe cast: we only modify last_direction, which is a mutable cache
        const_cast<Plan&>(plan).last_direction = direction;
    }

    // -- Launch kernel --------------------------------------------------------
    return launch_fft(
        reinterpret_cast<const float2*>(d_in),
        reinterpret_cast<float2*>(d_out),
        plan.N, plan.B, direction, stream
    );
}

// =============================================================================
// Plan Destroy
// =============================================================================

Status plan_destroy(Plan& plan) {
    // Free persisted host twiddle buffers
    if (plan.h_twiddle_fwd) {
        free(plan.h_twiddle_fwd);
        plan.h_twiddle_fwd = nullptr;
    }
    if (plan.h_twiddle_inv) {
        free(plan.h_twiddle_inv);
        plan.h_twiddle_inv = nullptr;
    }

    // Constant memory doesn't need explicit deallocation
    // Just reset the plan state
    plan.initialized = false;
    plan.N = 0;
    plan.B = 0;
    plan.last_direction = 0;
    return Status::SUCCESS;
}

// =============================================================================
// Expected GFLOP/s
// =============================================================================

double expected_gflops(const Plan& plan) {
    // Standard FFT FLOP count: 5 x N x log2(N) x B
    double flops = 5.0 * plan.N * plan.log2N * plan.B;
    return flops / 1e9;  // return in GFLOP (not GFLOP/s -- need timing for that)
}

} // namespace fft

// =============================================================================
// C API Exports (for Python ctypes / shared library)
// =============================================================================

extern "C" {

/**
 * @brief C-linkage: Create an FFT plan.
 *
 * Allocates a Plan on the heap, initializes it, and returns the handle.
 * The caller must eventually call fft_plan_destroy() to free resources.
 */
int fft_plan_create(void** plan_handle, int N, int B, int precision) {
    fft::Plan* plan = new (std::nothrow) fft::Plan();
    if (!plan) return static_cast<int>(fft::Status::ALLOC_FAILED);

    fft::Precision prec = (precision == 1) ? fft::Precision::F64
                                            : fft::Precision::F32;

    fft::Status status = fft::plan_create(*plan, N, B, prec);
    if (status != fft::Status::SUCCESS) {
        delete plan;
        *plan_handle = nullptr;
        return static_cast<int>(status);
    }

    *plan_handle = static_cast<void*>(plan);
    return 0;
}

/**
 * @brief C-linkage: Execute FFT transforms.
 */
int fft_plan_execute(void* plan_handle, void* d_in, void* d_out,
                     int direction) {
    if (!plan_handle) return static_cast<int>(fft::Status::NULL_POINTER);

    fft::Plan* plan = static_cast<fft::Plan*>(plan_handle);
    fft::Direction dir = (direction > 0) ? fft::Direction::INVERSE
                                          : fft::Direction::FORWARD;

    fft::Status status = fft::plan_execute(*plan, d_in, d_out, dir);
    if (status != fft::Status::SUCCESS) {
        return static_cast<int>(status);
    }

    // Synchronize to ensure results are available
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return static_cast<int>(fft::Status::EXECUTION_FAILED);
    }

    return 0;
}

/**
 * @brief C-linkage: Destroy an FFT plan and free resources.
 */
int fft_plan_destroy(void* plan_handle) {
    if (!plan_handle) return 0;
    fft::Plan* plan = static_cast<fft::Plan*>(plan_handle);
    fft::Status status = fft::plan_destroy(*plan);
    delete plan;
    return static_cast<int>(status);
}

/**
 * @brief C-linkage: Get expected GFLOP count.
 */
double fft_expected_gflops(void* plan_handle) {
    if (!plan_handle) return 0.0;
    fft::Plan* plan = static_cast<fft::Plan*>(plan_handle);
    return fft::expected_gflops(*plan);
}

} // extern "C"
