/**
 * @file freefloatfft_kernel.cuh
 * @brief Main FFT kernel template and launch dispatcher
 *
 * Matches Design Doc section 4.4 exactly. Execution phases per transform:
 *   Phase 1: Load with bit-reversal permutation (global -> register)
 *   Phase 2: Shared memory allocation (__shared__ smem_lo[N/2], smem_hi[N/2])
 *   Phase 3: Warp-synchronous stages (log2(warpSize) = 5 stages, no __syncthreads)
 *   Phase 4: Store to shared memory (odd/even split for bank conflict elimination)
 *   Phase 5: Cross-warp stages in shared memory (remaining stages)
 *   Phase 6: Read result and store to global memory (optional 1/N normalization)
 *
 * Each block handles one transform. Grid = B blocks, N threads/block.
 *
 * Template dispatch: compiler generates specialized code per N value.
 * Supported N: 64, 128, 256, 512, 1024, 2048.
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
 * @brief Main FreefloatFFT kernel: executes B transforms of length N in one launch.
 *
 * Design Doc section 4.4 — exact implementation.
 *
 * @tparam N          FFT length (64, 128, 256, 512, 1024, 2048)
 * @param  d_in       [B x N] complex input (global memory, float2*)
 * @param  d_out      [B x N] complex output (supports in-place: d_in == d_out)
 * @param  B          Batch count
 * @param  direction  -1 = forward FFT, +1 = inverse FFT
 */
template<int N>
__global__ void freefloatfft_kernel(
    const float2* __restrict__ d_in,   // [B x N] input
    float2*       __restrict__ d_out,  // [B x N] output
    int B,
    int direction   // -1 = forward, +1 = inverse
) {
    // Each block handles one transform
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_id >= B) return;

    // -- Phase 1: Load with bit-reversal permutation ----------------------------
    // Load element at bit-reversed position into registers
    int src_idx = c_bitrev[tid];  // bit-reversed index from constant memory
    float2 val = d_in[batch_id * N + src_idx];

    // -- Phase 2: Load into shared memory for cross-warp stages -----------------
    // Two SMEM arrays for bank-conflict-free butterfly access
    // Design Doc section 5.3: smem_hi offset by +1 for bank shift
    __shared__ float2 smem_lo[N / 2];
    __shared__ float2 smem_hi_store[N / 2 + 1];  // +1 shifts by 1 float2 = 2 banks
    float2* smem_hi = smem_hi_store + 1;           // skip first slot

    // -- Phase 3: Warp-synchronous stages (no __syncthreads) --------------------
    // Execute log2(warpSize) = 5 stages purely in registers + warp shuffles
    int lane = tid % warpSize;
    val = fft_warp_stages<(N < 32 ? N : 32)>(val, lane, N);

    // -- Phase 4: Store to shared memory after warp stages ----------------------
    // Odd/even split for bank conflict elimination
    bool is_hi = (tid >> (__ffs(warpSize) - 1)) & 1;  // bit at log2(warpSize)
    int smem_idx = tid & (N/2 - 1);

    if (is_hi) smem_hi[smem_idx] = val;
    else       smem_lo[smem_idx] = val;

    __syncthreads();

    // -- Phase 5: Cross-warp stages in shared memory ----------------------------
    fft_shared_stages<N>(smem_lo, smem_hi, tid);
    // Note: fft_shared_stages ends with __syncthreads()

    // -- Phase 6: Read result and store to global memory ------------------------
    // Reconstruct natural-order output
    bool final_hi = (tid >> (__ffs(N) - 2)) & 1;
    int final_idx = tid & (N/2 - 1);

    float2 result = final_hi ? smem_hi[final_idx] : smem_lo[final_idx];

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

/**
 * @brief Launch the appropriate kernel template for the given N.
 *
 * Template dispatch -- compiler generates specialized code per N.
 * Grid: B blocks, N threads per block (N <= 1024).
 * For N=2048: 1024 threads with 2 elements per thread (future enhancement).
 *
 * Matches Design Doc section 4.4 switch dispatch.
 */
inline fft::Status launch_fft(const float2* d_in, float2* d_out,
                               int N, int B, int direction,
                               cudaStream_t stream) {
    dim3 grid(B);
    dim3 block(N);

    // Template dispatch -- compiler generates specialized, fully-unrolled code per N
    switch (N) {
        case   64: freefloatfft_kernel< 64><<<grid, block, 0, stream>>>(d_in, d_out, B, direction); break;
        case  128: freefloatfft_kernel<128><<<grid, block, 0, stream>>>(d_in, d_out, B, direction); break;
        case  256: freefloatfft_kernel<256><<<grid, block, 0, stream>>>(d_in, d_out, B, direction); break;
        case  512: freefloatfft_kernel<512><<<grid, block, 0, stream>>>(d_in, d_out, B, direction); break;
        case 1024: freefloatfft_kernel<1024><<<grid,block, 0, stream>>>(d_in, d_out, B, direction); break;
        default:
            fprintf(stderr, "FreefloatFFT: unsupported N=%d -- must be power-of-2 in [64, 1024]\n", N);
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
