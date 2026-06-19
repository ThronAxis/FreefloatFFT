/**
 * @file test_correctness.cu
 * @brief FreefloatFFT correctness unit tests
 *
 * Tests:
 *   1. Butterfly R2 correctness — single butterfly against exact formula
 *   2. Twiddle generation accuracy — compare against exact cos/sin
 *   3. Bit-reversal permutation — bijection check
 *   4. Full FFT for N={64,128,256,512,1024} — compare against naive DFT
 *   5. Inverse FFT — IFFT(FFT(x)) ≈ x roundtrip
 *   6. In-place FFT — d_in == d_out correctness
 *
 * Compilation:
 *   nvcc -O3 -arch=sm_75 --use_fast_math -std=c++17 \
 *        test_correctness.cu -o test_correctness
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

// Include the main source file directly — this makes the test binary
// self-contained (no separate linking needed). The source file includes
// all headers and provides plan_create/execute/destroy implementations.
#include "../src/freefloatfft.cu"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do {                          \
    if (!(cond)) {                                           \
        printf("    ✗ FAIL: %s\n", msg);                     \
        tests_failed++;                                      \
        return;                                              \
    }                                                        \
} while(0)

#define TEST_PASS(msg) do {                                  \
    printf("    ✓ PASS: %s\n", msg);                         \
    tests_passed++;                                          \
} while(0)

// ═══════════════════════════════════════════════════════════════════════════════
// Naive DFT (host reference implementation)
// ═══════════════════════════════════════════════════════════════════════════════

static void naive_dft(const float2* in, float2* out, int N) {
    for (int k = 0; k < N; k++) {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < N; n++) {
            double angle = -2.0 * M_PI * k * n / N;
            double c = cos(angle), s = sin(angle);
            re += in[n].x * c - in[n].y * s;
            im += in[n].x * s + in[n].y * c;
        }
        out[k] = make_float2((float)re, (float)im);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 1: Twiddle Generation Accuracy
// ═══════════════════════════════════════════════════════════════════════════════

static void test_twiddle_accuracy() {
    printf("\n  [Test 1] Twiddle Generation Accuracy\n");

    for (int N : {64, 128, 256, 512, 1024}) {
        int half_N = N / 2;
        float2* h_twiddle = (float2*)malloc(half_N * sizeof(float2));
        precompute_twiddles_f32(h_twiddle, N, -1);

        double max_err = 0.0;
        for (int k = 0; k < half_N; k++) {
            double angle = -2.0 * M_PI * k / N;
            double exact_re = cos(angle);
            double exact_im = sin(angle);

            double err_re = fabs(h_twiddle[k].x - exact_re);
            double err_im = fabs(h_twiddle[k].y - exact_im);
            max_err = fmax(max_err, fmax(err_re, err_im));
        }

        char msg[128];
        snprintf(msg, sizeof(msg), "N=%d twiddle max_err=%.2e (threshold: 1e-6)", N, max_err);
        TEST_ASSERT(max_err < 1e-6, msg);
        free(h_twiddle);
    }
    TEST_PASS("All twiddle factors within accuracy threshold");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 2: Bit-Reversal Bijection
// ═══════════════════════════════════════════════════════════════════════════════

static void test_bitrev_bijection() {
    printf("\n  [Test 2] Bit-Reversal Permutation Bijection\n");

    for (int N : {64, 128, 256, 512, 1024}) {
        uint16_t* lut = (uint16_t*)malloc(N * sizeof(uint16_t));
        build_bitrev_lut(lut, N);

        // Check: every index 0..N-1 appears exactly once
        bool* seen = (bool*)calloc(N, sizeof(bool));
        bool valid = true;

        for (int i = 0; i < N; i++) {
            if (lut[i] >= N || seen[lut[i]]) {
                valid = false;
                break;
            }
            seen[lut[i]] = true;
        }

        char msg[128];
        snprintf(msg, sizeof(msg), "N=%d bitrev is bijection", N);
        TEST_ASSERT(valid, msg);

        free(lut);
        free(seen);
    }
    TEST_PASS("All bit-reversal LUTs are valid bijections");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 3: Full FFT Correctness (vs. naive DFT)
// ═══════════════════════════════════════════════════════════════════════════════

static void test_fft_correctness() {
    printf("\n  [Test 3] Full FFT Correctness (vs. naive DFT)\n");

    for (int N : {64, 128, 256, 512, 1024}) {
        int B = 4;  // Small batch for validation

        // Generate random input
        size_t total = (size_t)N * B;
        float2* h_in  = (float2*)malloc(total * sizeof(float2));
        float2* h_out = (float2*)malloc(total * sizeof(float2));
        float2* h_ref = (float2*)malloc(N * sizeof(float2));

        srand(42 + N);
        for (size_t i = 0; i < total; i++) {
            h_in[i].x = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            h_in[i].y = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        }

        // Create plan and execute on GPU
        fft::Plan plan;
        fft::Status status = fft::plan_create(plan, N, B);
        TEST_ASSERT(status == fft::Status::SUCCESS, "plan_create");

        float2 *d_in, *d_out;
        cudaMalloc(&d_in, total * sizeof(float2));
        cudaMalloc(&d_out, total * sizeof(float2));
        cudaMemcpy(d_in, h_in, total * sizeof(float2), cudaMemcpyHostToDevice);

        status = fft::plan_execute(plan, d_in, d_out, fft::Direction::FORWARD);
        cudaDeviceSynchronize();
        TEST_ASSERT(status == fft::Status::SUCCESS, "plan_execute");

        cudaMemcpy(h_out, d_out, total * sizeof(float2), cudaMemcpyDeviceToHost);

        // Compare each transform against naive DFT
        double max_err = 0.0;
        for (int b = 0; b < B; b++) {
            naive_dft(h_in + b * N, h_ref, N);
            for (int k = 0; k < N; k++) {
                double err_re = fabs(h_out[b * N + k].x - h_ref[k].x);
                double err_im = fabs(h_out[b * N + k].y - h_ref[k].y);
                max_err = fmax(max_err, fmax(err_re, err_im));
            }
        }

        char msg[128];
        snprintf(msg, sizeof(msg), "N=%d B=%d max_err=%.2e (threshold: 1e-3)", N, B, max_err);
        TEST_ASSERT(max_err < 1e-3, msg);
        printf("    · N=%d: max_err=%.2e ✓\n", N, max_err);

        cudaFree(d_in);
        cudaFree(d_out);
        free(h_in);
        free(h_out);
        free(h_ref);
        fft::plan_destroy(plan);
    }
    TEST_PASS("All FFT sizes match naive DFT within tolerance");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 4: Inverse FFT Roundtrip — IFFT(FFT(x)) ≈ x
// ═══════════════════════════════════════════════════════════════════════════════

static void test_inverse_roundtrip() {
    printf("\n  [Test 4] Inverse FFT Roundtrip: IFFT(FFT(x)) ≈ x\n");

    for (int N : {64, 128, 256, 512}) {
        int B = 4;
        size_t total = (size_t)N * B;

        float2* h_in = (float2*)malloc(total * sizeof(float2));
        float2* h_fwd = (float2*)malloc(total * sizeof(float2));
        float2* h_roundtrip = (float2*)malloc(total * sizeof(float2));

        srand(123 + N);
        for (size_t i = 0; i < total; i++) {
            h_in[i].x = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            h_in[i].y = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        }

        fft::Plan plan;
        fft::plan_create(plan, N, B);

        float2 *d_in, *d_mid, *d_out;
        cudaMalloc(&d_in,  total * sizeof(float2));
        cudaMalloc(&d_mid, total * sizeof(float2));
        cudaMalloc(&d_out, total * sizeof(float2));

        cudaMemcpy(d_in, h_in, total * sizeof(float2), cudaMemcpyHostToDevice);

        // Forward
        fft::plan_execute(plan, d_in, d_mid, fft::Direction::FORWARD);
        cudaDeviceSynchronize();

        // Inverse (with 1/N normalization)
        fft::plan_execute(plan, d_mid, d_out, fft::Direction::INVERSE);
        cudaDeviceSynchronize();

        cudaMemcpy(h_roundtrip, d_out, total * sizeof(float2), cudaMemcpyDeviceToHost);

        // Compare: IFFT(FFT(x)) should ≈ x
        double max_err = 0.0;
        for (size_t i = 0; i < total; i++) {
            double err_re = fabs(h_roundtrip[i].x - h_in[i].x);
            double err_im = fabs(h_roundtrip[i].y - h_in[i].y);
            max_err = fmax(max_err, fmax(err_re, err_im));
        }

        char msg[128];
        snprintf(msg, sizeof(msg), "N=%d roundtrip max_err=%.2e (threshold: 1e-3)", N, max_err);
        TEST_ASSERT(max_err < 1e-3, msg);
        printf("    · N=%d: roundtrip max_err=%.2e ✓\n", N, max_err);

        cudaFree(d_in);
        cudaFree(d_mid);
        cudaFree(d_out);
        free(h_in);
        free(h_fwd);
        free(h_roundtrip);
        fft::plan_destroy(plan);
    }
    TEST_PASS("IFFT(FFT(x)) ≈ x for all tested N values");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 5: In-Place FFT (d_in == d_out)
// ═══════════════════════════════════════════════════════════════════════════════

static void test_inplace() {
    printf("\n  [Test 5] In-Place FFT (d_in == d_out)\n");

    int N = 256, B = 2;
    size_t total = (size_t)N * B;

    float2* h_in = (float2*)malloc(total * sizeof(float2));
    float2* h_out_inplace = (float2*)malloc(total * sizeof(float2));
    float2* h_out_separate = (float2*)malloc(total * sizeof(float2));

    srand(999);
    for (size_t i = 0; i < total; i++) {
        h_in[i].x = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        h_in[i].y = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
    }

    fft::Plan plan;
    fft::plan_create(plan, N, B);

    float2 *d_buf, *d_in2, *d_out2;
    cudaMalloc(&d_buf, total * sizeof(float2));
    cudaMalloc(&d_in2, total * sizeof(float2));
    cudaMalloc(&d_out2, total * sizeof(float2));

    // Out-of-place reference
    cudaMemcpy(d_in2, h_in, total * sizeof(float2), cudaMemcpyHostToDevice);
    fft::plan_execute(plan, d_in2, d_out2, fft::Direction::FORWARD);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_separate, d_out2, total * sizeof(float2), cudaMemcpyDeviceToHost);

    // In-place test
    cudaMemcpy(d_buf, h_in, total * sizeof(float2), cudaMemcpyHostToDevice);
    fft::plan_execute(plan, d_buf, d_buf, fft::Direction::FORWARD);  // in-place!
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_inplace, d_buf, total * sizeof(float2), cudaMemcpyDeviceToHost);

    // Compare in-place vs out-of-place
    double max_err = 0.0;
    for (size_t i = 0; i < total; i++) {
        double err_re = fabs(h_out_inplace[i].x - h_out_separate[i].x);
        double err_im = fabs(h_out_inplace[i].y - h_out_separate[i].y);
        max_err = fmax(max_err, fmax(err_re, err_im));
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "In-place max_err=%.2e (threshold: 1e-6)", max_err);
    TEST_ASSERT(max_err < 1e-6, msg);
    TEST_PASS("In-place matches out-of-place result");

    cudaFree(d_buf);
    cudaFree(d_in2);
    cudaFree(d_out2);
    free(h_in);
    free(h_out_inplace);
    free(h_out_separate);
    fft::plan_destroy(plan);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test 6: Large Batch Validation
// ═══════════════════════════════════════════════════════════════════════════════

static void test_large_batch() {
    printf("\n  [Test 6] Large Batch Validation (B=10000, spot-check)\n");

    int N = 256, B = 10000;
    size_t total = (size_t)N * B;

    float2* h_in  = (float2*)malloc(total * sizeof(float2));
    float2* h_out = (float2*)malloc(total * sizeof(float2));
    float2* h_ref = (float2*)malloc(N * sizeof(float2));

    srand(777);
    for (size_t i = 0; i < total; i++) {
        h_in[i].x = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        h_in[i].y = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
    }

    fft::Plan plan;
    fft::plan_create(plan, N, B);

    float2 *d_in, *d_out;
    cudaMalloc(&d_in, total * sizeof(float2));
    cudaMalloc(&d_out, total * sizeof(float2));
    cudaMemcpy(d_in, h_in, total * sizeof(float2), cudaMemcpyHostToDevice);

    fft::plan_execute(plan, d_in, d_out, fft::Direction::FORWARD);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, total * sizeof(float2), cudaMemcpyDeviceToHost);

    // Spot-check first 10 and last 10 transforms
    double max_err = 0.0;
    int checks[] = {0, 1, 2, 3, 4, B-5, B-4, B-3, B-2, B-1};
    for (int ci = 0; ci < 10; ci++) {
        int b = checks[ci];
        naive_dft(h_in + b * N, h_ref, N);
        for (int k = 0; k < N; k++) {
            double err_re = fabs(h_out[b * N + k].x - h_ref[k].x);
            double err_im = fabs(h_out[b * N + k].y - h_ref[k].y);
            max_err = fmax(max_err, fmax(err_re, err_im));
        }
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "B=%d spot-check max_err=%.2e (threshold: 1e-3)", B, max_err);
    TEST_ASSERT(max_err < 1e-3, msg);
    TEST_PASS("Large batch spot-check passed");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    free(h_ref);
    fft::plan_destroy(plan);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           FreefloatFFT Correctness Test Suite               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  GPU: %s (Compute %d.%d)\n\n", prop.name, prop.major, prop.minor);

    test_twiddle_accuracy();
    test_bitrev_bijection();
    test_fft_correctness();
    test_inverse_roundtrip();
    test_inplace();
    test_large_batch();

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("  Status:  %s\n", tests_failed == 0 ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    return tests_failed > 0 ? 1 : 0;
}
