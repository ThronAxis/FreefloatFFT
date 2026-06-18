/**
 * @file freefloatfft_math.cuh
 * @brief Complex arithmetic primitives for FreefloatFFT
 *
 * All operations are __device__ __forceinline__ — zero function call overhead.
 * All operations are register-only — zero memory traffic.
 * Complex number representation: float2 where .x = real, .y = imaginary.
 */

#ifndef FREEFLOATFFT_MATH_CUH
#define FREEFLOATFFT_MATH_CUH

#include <cuda_runtime.h>

/** @brief Complex addition: c = a + b (2 FADD) */
__device__ __forceinline__
float2 cadd(const float2 a, const float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

/** @brief Complex subtraction: c = a - b (2 FSUB) */
__device__ __forceinline__
float2 csub(const float2 a, const float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

/**
 * @brief Complex multiplication using FMA: c = a × b (2 FMA + 2 FMUL = 6 FP ops)
 * real = fma(a.x, b.x, -(a.y*b.y)), imag = fma(a.x, b.y, a.y*b.x)
 */
__device__ __forceinline__
float2 cmul(const float2 a, const float2 b) {
    return make_float2(
        __fmaf_rn(a.x, b.x, -(a.y * b.y)),
        __fmaf_rn(a.x, b.y,   a.y * b.x)
    );
}

/** @brief Multiply by -j: (-i)(a+ib) = (b, -a). Zero FP ops — register rename. */
__device__ __forceinline__
float2 cmul_neg_j(const float2 a) {
    return make_float2(a.y, -a.x);
}

/** @brief Multiply by +j: (i)(a+ib) = (-b, a). Zero FP ops. */
__device__ __forceinline__
float2 cmul_pos_j(const float2 a) {
    return make_float2(-a.y, a.x);
}

/** @brief Complex conjugate: conj(a) = (a.x, -a.y) */
__device__ __forceinline__
float2 cconj(const float2 a) {
    return make_float2(a.x, -a.y);
}

/** @brief Scale complex by real scalar: c = s × a */
__device__ __forceinline__
float2 cscale(const float2 a, float s) {
    return make_float2(a.x * s, a.y * s);
}

/**
 * @brief Radix-2 DIT butterfly: (a, b) → (a + W·b, a − W·b)
 * Total: 10 FP ops (cmul=6 + cadd=2 + csub=2)
 */
__device__ __forceinline__
void butterfly(float2& a, float2& b, const float2 W) {
    float2 t = cmul(W, b);
    b = csub(a, t);
    a = cadd(a, t);
}

/**
 * @brief Radix-4 DIT butterfly: processes 4 elements, halves stage count.
 * 8 complex add + 3 complex multiply = 34 FP ops
 */
__device__ __forceinline__
void butterfly_r4(float2& x0, float2& x1, float2& x2, float2& x3,
                  const float2 w1, const float2 w2, const float2 w3) {
    float2 t0 = cadd(x0, x2);
    float2 t2 = csub(x0, x2);
    float2 t1 = cadd(x1, x3);
    float2 t3 = csub(x1, x3);
    t3 = cmul_neg_j(t3);
    x0 = cadd(t0, t1);
    x1 = cmul(csub(t0, t1), w1);
    x2 = cmul(cadd(t2, t3), w2);
    x3 = cmul(csub(t2, t3), w3);
}

#endif // FREEFLOATFFT_MATH_CUH
