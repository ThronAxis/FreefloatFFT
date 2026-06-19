/**
 * @file freefloatfft_kernel.cuh
 * @brief Main FFT kernel template and launch dispatcher
 *
 * Execution phases per transform:
 *   Phase 1: Load with bit-reversal permutation (global -> register)
 *   Phase 2: Warp-synchronous stages (log2(32) = 5 stages, no barriers)
 *   Phase 3: Store to shared memory (natural order)
 *   Phase 4: Cross-warp stages in shared memory (XOR butterfly with barriers)
 *   Phase 5: Read from shared memory -> global store (optional 1/N norm)
 *
 * Each block handles one transform. Grid = B blocks, N threads/block.
 * Template dispatch: compiler generates specialized code per N value.
 */

#ifndef FREEFLOATFFT_KERNEL_CUH
#define FREEFLOATFFT_KERNEL_CUH

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include "freefloatfft_math.cuh"
#include "freefloatfft_constants.cuh"
#include "freefloatfft_warp.cuh"
#include "freefloatfft_smem.cuh"
#include "freefloatfft.h"

/**
 * @brief Main FreefloatFFT kernel.
 *
 * @tparam N          FFT length (64, 128, 256, 512, 1024)
 * @param  d_in       [B x N] complex input
 * @param  d_out      [B x N] complex output (supports in-place)
 * @param  B          Batch count
 * @param  direction  -1 = forward FFT, +1 = inverse FFT
 */
template<int N>
__global__ void freefloatfft_kernel(
    const float2* __restrict__ d_in,
    float2*       __restrict__ d_out,
    int B,
    int direction
) {
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_id >= B) return;

    // -- Phase 1: Load with bit-reversal permutation --------------------------
    int src_idx = c_bitrev[tid];
    float2 val = d_in[batch_id * N + src_idx];

    // -- Phase 2: Warp-synchronous stages (no barriers) -----------------------
    int lane = tid % warpSize;
    val = fft_warp_stages<(N < 32 ? N : 32)>(val, lane, N);

    // -- Phase 3: Store to shared memory --------------------------------------
    extern __shared__ float2 smem[];
    smem[tid] = val;
    __syncthreads();

    // -- Phase 4: Cross-warp stages in shared memory --------------------------
    if (N > 32) {
        fft_shared_stages<N>(smem, tid);
    }

    // -- Phase 5: Read result and store to global memory ----------------------
    float2 result = smem[tid];

    // Optional: 1/N normalization for inverse transform
    if (direction > 0) {
        float inv_N = 1.0f / N;
        result.x *= inv_N;
        result.y *= inv_N;
    }

    d_out[batch_id * N + tid] = result;
}

// =============================================================================
// Kernel Launch Dispatcher
// =============================================================================

inline fft::Status launch_fft(const float2* d_in, float2* d_out,
                               int N, int B, int direction,
                               cudaStream_t stream) {
    dim3 grid(B);
    dim3 block(N);
    size_t smem_bytes = N * sizeof(float2);

    switch (N) {
        case   64: freefloatfft_kernel< 64><<<grid, block, smem_bytes, stream>>>(d_in, d_out, B, direction); break;
        case  128: freefloatfft_kernel<128><<<grid, block, smem_bytes, stream>>>(d_in, d_out, B, direction); break;
        case  256: freefloatfft_kernel<256><<<grid, block, smem_bytes, stream>>>(d_in, d_out, B, direction); break;
        case  512: freefloatfft_kernel<512><<<grid, block, smem_bytes, stream>>>(d_in, d_out, B, direction); break;
        case 1024: freefloatfft_kernel<1024><<<grid,block, smem_bytes, stream>>>(d_in, d_out, B, direction); break;
        default:
            fprintf(stderr, "FreefloatFFT: unsupported N=%d\n", N);
            return fft::Status::INVALID_N;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "FreefloatFFT kernel launch error: %s\n",
                cudaGetErrorString(err));
        return fft::Status::EXECUTION_FAILED;
    }

    return fft::Status::SUCCESS;
}

#endif // FREEFLOATFFT_KERNEL_CUH
