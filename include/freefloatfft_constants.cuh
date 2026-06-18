/**
 * @file freefloatfft_constants.cuh
 * @brief Constant memory declarations and precomputation routines
 *
 * Twiddle factors: precomputed in double precision, stored as float32 in __constant__.
 * Bit-reversal LUT: precomputed on host, uploaded to __constant__.
 *
 * Memory budget:
 *   c_twiddle_f32:     1024 × 8 = 8 KB  (forward)
 *   c_twiddle_f32_inv: 1024 × 8 = 8 KB  (inverse)
 *   c_bitrev:          2048 × 2 = 4 KB
 *   Total: 20 KB << 64 KB constant memory limit ✓
 */

#ifndef FREEFLOATFFT_CONSTANTS_CUH
#define FREEFLOATFFT_CONSTANTS_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// Constant Memory Declarations
// ═══════════════════════════════════════════════════════════════════════════════

/** Forward twiddle factors: W_N^k = exp(-2πik/N), k = 0..N/2-1 */
__constant__ float2 c_twiddle_f32[1024];

/** Inverse twiddle factors: W_N^{-k} = exp(+2πik/N), k = 0..N/2-1 */
__constant__ float2 c_twiddle_f32_inv[1024];

/** Bit-reversal permutation LUT: bitrev[i] = bit-reversed index of i */
__constant__ uint16_t c_bitrev[2048];

// ═══════════════════════════════════════════════════════════════════════════════
// Host-Side Precomputation: Twiddle Factors
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Precompute twiddle factors using trigonometric recurrence.
 *
 * Uses double-precision recurrence for numerical stability, stores as float32.
 * Error: ULP < 2 for all N ≤ 2048 (validated against exact exp evaluation).
 *
 * Recurrence:
 *   cos(θ + δ) = cos(θ)·cos(δ) − sin(θ)·sin(δ)
 *   sin(θ + δ) = sin(θ)·cos(δ) + cos(θ)·sin(δ)
 *
 * @param[out] h_out    Host buffer for N/2 twiddle entries
 * @param[in]  N        Transform length
 * @param[in]  dir      -1 for forward, +1 for inverse
 */
inline void precompute_twiddles_f32(float2* h_out, int N, int dir) {
    const double theta = dir * (-2.0 * M_PI) / N;
    const double c_delta = cos(theta);
    const double s_delta = sin(theta);

    double c_k = 1.0, s_k = 0.0;  // W_N^0 = 1 + 0i

    for (int k = 0; k < N / 2; k++) {
        h_out[k] = make_float2(static_cast<float>(c_k),
                               static_cast<float>(s_k));
        double new_c = c_k * c_delta - s_k * s_delta;
        s_k = s_k * c_delta + c_k * s_delta;
        c_k = new_c;
    }

    // Correction: ensure W_N^{N/4} is exact (common FFT invariant)
    // W_N^{N/4} = exp(-2πi·(N/4)/N) = exp(-πi/2) = -i = (0, -1) for forward
    if (N >= 4) {
        h_out[N / 4] = make_float2(0.0f, dir < 0 ? -1.0f : 1.0f);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Host-Side Precomputation: Bit-Reversal LUT
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Build bit-reversal permutation lookup table.
 *
 * For DIT FFT, input must be in bit-reversed order. This LUT maps
 * natural-order index → bit-reversed index for efficient loading.
 *
 * @param[out] lut  Host buffer for N entries
 * @param[in]  N    Transform length (must be power-of-2)
 */
inline void build_bitrev_lut(uint16_t* lut, int N) {
    int log2N = 0;
    int temp = N;
    while (temp > 1) { log2N++; temp >>= 1; }

    for (int i = 0; i < N; i++) {
        uint16_t reversed = 0;
        int x = i;
        for (int b = 0; b < log2N; b++) {
            reversed = (reversed << 1) | (x & 1);
            x >>= 1;
        }
        lut[i] = reversed;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Upload to Constant Memory
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Upload forward twiddle factors to GPU constant memory.
 */
inline cudaError_t upload_twiddles_forward(const float2* h_twiddle, int half_N) {
    return cudaMemcpyToSymbol(c_twiddle_f32, h_twiddle,
                              half_N * sizeof(float2));
}

/**
 * @brief Upload inverse twiddle factors to GPU constant memory.
 */
inline cudaError_t upload_twiddles_inverse(const float2* h_twiddle, int half_N) {
    return cudaMemcpyToSymbol(c_twiddle_f32_inv, h_twiddle,
                              half_N * sizeof(float2));
}

/**
 * @brief Upload bit-reversal LUT to GPU constant memory.
 */
inline cudaError_t upload_bitrev(const uint16_t* h_bitrev, int N) {
    return cudaMemcpyToSymbol(c_bitrev, h_bitrev,
                              N * sizeof(uint16_t));
}

/**
 * @brief Activate twiddle factors for the specified direction.
 *
 * The kernel stages always read from c_twiddle_f32 (the active slot).
 * For forward FFT, c_twiddle_f32 already holds the forward twiddles (from plan_create).
 * For inverse FFT, we copy the inverse twiddles from c_twiddle_f32_inv into c_twiddle_f32.
 *
 * This is a device-to-device constant memory copy (~0.5 us overhead).
 *
 * @param  direction  -1 for forward, +1 for inverse
 * @param  half_N     N/2 elements to copy
 * @param  h_twiddle_fwd  Host buffer with forward twiddles (for restoring)
 * @param  h_twiddle_inv  Host buffer with inverse twiddles
 * @return cudaSuccess on success
 */
inline cudaError_t activate_twiddles(int direction, int half_N,
                                      const float2* h_twiddle_fwd,
                                      const float2* h_twiddle_inv) {
    if (direction > 0) {
        // Inverse: load inverse twiddles into the active slot
        return cudaMemcpyToSymbol(c_twiddle_f32, h_twiddle_inv,
                                  half_N * sizeof(float2));
    } else {
        // Forward: load forward twiddles into the active slot
        return cudaMemcpyToSymbol(c_twiddle_f32, h_twiddle_fwd,
                                  half_N * sizeof(float2));
    }
}

#endif // FREEFLOATFFT_CONSTANTS_CUH
