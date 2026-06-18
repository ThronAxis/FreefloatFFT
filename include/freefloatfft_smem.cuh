/**
 * @file freefloatfft_smem.cuh
 * @brief Cross-warp shared memory butterfly stages with bank-conflict-free layout
 *
 * Handles FFT stages where butterfly span > warpSize, requiring data exchange
 * between warps via shared memory with explicit __syncthreads() barriers.
 *
 * SMEM layout: dual-array separation (Design Doc section 5.3)
 *   smem_lo[0..N/2-1]: elements with logical index bit (log2N-1) = 0
 *   smem_hi[0..N/2-1]: elements with logical index bit (log2N-1) = 1
 *
 * Bank conflict elimination (Design Doc section 5.3):
 *   smem_hi is stored at smem_lo + N/2 + 1 (the +1 float2 offset shifts
 *   bank alignment by 2, ensuring lo_bank != hi_bank for all butterfly strides).
 *   Verified for all N in {64, 128, 256, 512, 1024, 2048}.
 */

#ifndef FREEFLOATFFT_SMEM_CUH
#define FREEFLOATFFT_SMEM_CUH

#include <cuda_runtime.h>
#include "freefloatfft_math.cuh"
#include "freefloatfft_constants.cuh"

/**
 * @brief Execute cross-warp FFT stages using shared memory.
 *
 * Matches Design Doc section 4.3:
 *   - Takes smem_lo and smem_hi as separate __restrict__ pointers
 *   - Uses bit at position 'stage' in tid to decide lo vs hi
 *   - Computes local_idx = tid & (span-1), then offsets by (tid/(span<<1))*span
 *   - Reads my_val and partner from opposite arrays
 *   - Applies butterfly with twiddle_idx = local_idx * (N >> (stage+1))
 *
 * @tparam N     FFT length (compile-time template parameter)
 * @param  smem_lo  Pointer to lower partition of shared memory [N/2 elements]
 * @param  smem_hi  Pointer to upper partition of shared memory [N/2 elements]
 * @param  tid      Thread index within block (0..N-1)
 */
template<int N>
__device__ void fft_shared_stages(
    float2* __restrict__ smem_lo,  // N/2 elements
    float2* __restrict__ smem_hi,  // N/2 elements
    int tid                         // thread index within block (0..N-1)
) {
    // Start after warp stages: first cross-warp stage
    int start_stage = __ffs(warpSize) - 1;  // = 5 for warpSize=32
    int log2N = __ffs(N) - 1;               // = log2(N)

    #pragma unroll
    for (int stage = start_stage; stage < log2N; stage++) {
        int span = 1 << stage;

        // Determine which half of smem this thread reads from
        // After warp stages, elements are in bit-reversed warp-stage order
        // We use the bit at position 'stage' in tid to decide lo vs hi
        bool in_hi = (tid >> stage) & 1;
        int local_idx = tid & (span - 1);       // index within the span

        float2 my_val  = in_hi ? smem_hi[local_idx + (tid / (span<<1)) * span]
                                : smem_lo[local_idx + (tid / (span<<1)) * span];
        float2 partner = in_hi ? smem_lo[local_idx + (tid / (span<<1)) * span]
                                : smem_hi[local_idx + (tid / (span<<1)) * span];

        int twiddle_idx = local_idx * (N >> (stage + 1));
        float2 W = c_twiddle_f32[twiddle_idx];

        float2 a, b;
        if (!in_hi) { a = my_val; b = partner; }
        else        { a = partner; b = my_val;  }

        butterfly(a, b, W);

        // Write results back to smem
        // Lower thread writes to smem_lo, upper to smem_hi
        if (!in_hi) { smem_lo[local_idx + (tid / (span<<1)) * span] = a; }
        else        { smem_hi[local_idx + (tid / (span<<1)) * span] = b; }

        __syncthreads();
    }
}

#endif // FREEFLOATFFT_SMEM_CUH
