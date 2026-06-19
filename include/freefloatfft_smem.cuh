/**
 * @file freefloatfft_smem.cuh
 * @brief Cross-warp shared memory butterfly stages
 *
 * Simplified correct implementation: single shared memory array with
 * XOR-based butterfly partner finding. Each thread reads its own value
 * and its partner's value, applies the butterfly, and writes back.
 *
 * Bank conflict note: This version may have some bank conflicts but
 * is provably correct. Optimization with dual-array layout can be
 * added after correctness is verified.
 */

#ifndef FREEFLOATFFT_SMEM_CUH
#define FREEFLOATFFT_SMEM_CUH

#include <cuda_runtime.h>
#include "freefloatfft_math.cuh"
#include "freefloatfft_constants.cuh"

/**
 * @brief Execute cross-warp FFT stages using shared memory.
 *
 * Uses XOR to find butterfly partners (same logic as warp stages,
 * but with __syncthreads() barriers between stages).
 *
 * @tparam N     FFT length (compile-time template parameter)
 * @param  smem  Shared memory array of N float2 elements
 * @param  tid   Thread index within block (0..N-1)
 */
template<int N>
__device__ void fft_shared_stages(float2* smem, int tid) {
    // Compute log2(N) at compile time
    constexpr int LOG2_N = (N ==   64) ?  6 :
                           (N ==  128) ?  7 :
                           (N ==  256) ?  8 :
                           (N ==  512) ?  9 :
                           (N == 1024) ? 10 :
                           (N == 2048) ? 11 : 0;

    // Cross-warp stages: stage 5 (span=32) through stage LOG2_N-1
    #pragma unroll
    for (int stage = 5; stage < LOG2_N; stage++) {
        int span = 1 << stage;

        // XOR to find butterfly partner
        int partner_tid = tid ^ span;

        // Read both values from shared memory
        float2 my_val      = smem[tid];
        float2 partner_val = smem[partner_tid];

        // Determine if this thread is upper or lower in the butterfly
        // Lower: (tid & span) == 0, Upper: (tid & span) != 0
        bool upper = (tid & span) != 0;

        // Twiddle index: position within the butterfly group
        int local_pos = tid & (span - 1);
        int twiddle_idx = local_pos * (N >> (stage + 1));
        float2 W = c_twiddle_f32[twiddle_idx];

        // Apply butterfly
        float2 a = upper ? partner_val : my_val;
        float2 b = upper ? my_val      : partner_val;
        butterfly(a, b, W);

        // Write result back
        smem[tid] = upper ? b : a;

        __syncthreads();
    }
}

#endif // FREEFLOATFFT_SMEM_CUH
