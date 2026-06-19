/**
 * @file freefloatfft_warp.cuh
 * @brief Warp-synchronous butterfly stages using __shfl_xor_sync
 *
 * Handles FFT stages where butterfly span <= warpSize (stages 0 to log2(32)-1 = 4).
 * Uses __shfl_xor_sync to exchange values between warp lanes -- zero shared memory
 * traffic, zero __syncthreads() barriers.
 *
 * Performance advantage (§2.2 of Design Doc):
 *   __shfl_xor_sync:  ~4 cycles
 *   Shared mem R/W:   ~32+ cycles (no conflict), ~640 cycles (32-way conflict)
 *   __syncthreads():  ~25-800 cycles
 *
 * For N=512, this saves 5 __syncthreads() barriers x B transforms.
 *
 * Twiddle access pattern (§3.3):
 *   twiddle_idx = (pos & (span-1)) * (N_full >> (stage+1))
 *   All threads in a warp access same/nearby twiddle entries -> L1 broadcast.
 */

#ifndef FREEFLOATFFT_WARP_CUH
#define FREEFLOATFFT_WARP_CUH

#include <cuda_runtime.h>
#include "freefloatfft_math.cuh"
#include "freefloatfft_constants.cuh"

/**
 * @brief Execute all warp-synchronous FFT stages for thread's element 'val'.
 *
 * Assumes: all threads in warp hold one element of one transform.
 * Warp lane 'lane' corresponds to logical FFT index 'lane'.
 *
 * Matches Design Doc section 4.2 exactly:
 *   - XOR-based partner finding
 *   - Twiddle index = (pos & (span-1)) * (N_full >> (stage+1))
 *   - Lower branch: a = val, b = partner
 *   - Upper branch: a = partner, b = val
 *
 * @tparam N_WARP  min(N, warpSize) = elements per warp
 * @param  val     This thread's complex element (updated in-place)
 * @param  lane    Warp lane ID (threadIdx.x % 32)
 * @param  N_full  Full FFT length N (for twiddle index calculation)
 * @return Updated complex element after all warp stages
 */
template<int N_WARP>
__device__ __forceinline__
float2 fft_warp_stages(float2 val, int lane, int N_full) {
    // Number of warp stages = log2(N_WARP)
    // For N=512, warpSize=32: 5 warp stages

    #pragma unroll
    for (int stage = 0; (1 << stage) < N_WARP; stage++) {
        int span = 1 << stage;                    // butterfly span
        int group_size = span << 1;               // 2 x span
        int pos   = lane % group_size;            // position within group
        bool upper = pos >= span;                  // upper or lower branch

        // Twiddle index for this butterfly
        int twiddle_idx = (pos & (span - 1)) * (N_full >> (stage + 1));
        float2 W = c_twiddle_f32[twiddle_idx];

        // Exchange values with butterfly partner via warp shuffle
        float partner_x = __shfl_xor_sync(0xFFFFFFFF, val.x, span);
        float partner_y = __shfl_xor_sync(0xFFFFFFFF, val.y, span);
        float2 partner = make_float2(partner_x, partner_y);

        // Apply butterfly (lower branch: a = val, b = partner)
        //                  (upper branch: a = partner, b = val)
        float2 a = upper ? partner : val;
        float2 b = upper ? val     : partner;

        butterfly(a, b, W);

        val = upper ? b : a;  // take the correct output
    }
    return val;
}

#endif // FREEFLOATFFT_WARP_CUH
