/**
 * @file freefloatfft_bench.cu
 * @brief Built-in benchmarking suite for FreefloatFFT
 *
 * Standalone binary that benchmarks FreefloatFFT across all supported N values
 * and various batch sizes. Reports throughput in GFLOP/s and milliseconds.
 *
 * Usage:
 *   ./freefloatfft_bench [--N <size>] [--B <batch>] [--iters <count>]
 *
 * Compilation:
 *   nvcc -O3 -arch=sm_75 --use_fast_math -std=c++17 \
 *        freefloatfft_bench.cu -o freefloatfft_bench
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#include "../include/freefloatfft.h"
#include "../include/freefloatfft_math.cuh"
#include "../include/freefloatfft_constants.cuh"
#include "../include/freefloatfft_warp.cuh"
#include "../include/freefloatfft_smem.cuh"
#include "../include/freefloatfft_kernel.cuh"

// ═══════════════════════════════════════════════════════════════════════════════
// Random Complex Signal Generator
// ═══════════════════════════════════════════════════════════════════════════════

static void generate_random_complex(float2* h_data, int count, unsigned seed) {
    srand(seed);
    for (int i = 0; i < count; i++) {
        h_data[i].x = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        h_data[i].y = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Runner
// ═══════════════════════════════════════════════════════════════════════════════

static void benchmark_fft(int N, int B, int iters) {
    // ── Create plan ──
    fft::Plan plan;
    fft::Status status = fft::plan_create(plan, N, B);
    if (status != fft::Status::SUCCESS) {
        fprintf(stderr, "BENCH: plan_create failed for N=%d, B=%d\n", N, B);
        return;
    }

    // ── Allocate device memory ──
    size_t total_elements = (size_t)N * B;
    size_t bytes = total_elements * sizeof(float2);

    float2* d_in = nullptr;
    float2* d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    if (!d_in || !d_out) {
        fprintf(stderr, "BENCH: cudaMalloc failed (%zu MB)\n", bytes / (1024 * 1024));
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        return;
    }

    // ── Generate and upload random input ──
    float2* h_data = (float2*)malloc(bytes);
    generate_random_complex(h_data, total_elements, 42);
    cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice);
    free(h_data);

    // ── Warmup ──
    for (int i = 0; i < 3; i++) {
        fft::plan_execute(plan, d_in, d_out, fft::Direction::FORWARD);
    }
    cudaDeviceSynchronize();

    // ── Timed runs ──
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        fft::plan_execute(plan, d_in, d_out, fft::Direction::FORWARD);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / iters;

    // ── Compute metrics ──
    double flop_count = 5.0 * N * log2((double)N) * B;
    double gflops = (flop_count / (avg_ms * 1e-3)) / 1e9;
    double throughput_transforms = B / (avg_ms * 1e-3);

    printf("  N=%5d  B=%8d | %8.3f ms | %7.1f GFLOP/s | %10.0f transforms/s\n",
           N, B, avg_ms, gflops, throughput_transforms);

    // ── Cleanup ──
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    fft::plan_destroy(plan);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    // Parse optional arguments
    int single_N = 0, single_B = 0, iters = 20;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--N") == 0 && i + 1 < argc) {
            single_N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--B") == 0 && i + 1 < argc) {
            single_B = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        }
    }

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║            FreefloatFFT Benchmark Suite v1.0               ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  GPU: %-52s ║\n", prop.name);
    printf("║  SMs: %-3d   Compute: %d.%d   VRAM: %zu MB              ║\n",
           prop.multiProcessorCount, prop.major, prop.minor,
           prop.totalGlobalMem / (1024 * 1024));
    printf("║  Iterations per measurement: %-3d                           ║\n", iters);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    if (single_N > 0 && single_B > 0) {
        // Single configuration
        printf("  %-7s %-10s | %10s | %13s | %18s\n",
               "N", "Batch", "Time", "Throughput", "Transforms/s");
        printf("  ──────────────────┼────────────┼───────────────┼──────────────────\n");
        benchmark_fft(single_N, single_B, iters);
    } else {
        // Full sweep
        int Ns[] = {64, 128, 256, 512, 1024};
        int Bs[] = {1000, 10000, 100000};

        printf("  %-7s %-10s | %10s | %13s | %18s\n",
               "N", "Batch", "Time", "Throughput", "Transforms/s");
        printf("  ──────────────────┼────────────┼───────────────┼──────────────────\n");

        for (int ni = 0; ni < 5; ni++) {
            for (int bi = 0; bi < 3; bi++) {
                // Check if we have enough GPU memory
                size_t needed = (size_t)Ns[ni] * Bs[bi] * sizeof(float2) * 2;
                size_t free_mem, total_mem;
                cudaMemGetInfo(&free_mem, &total_mem);
                if (needed > free_mem * 0.8) {
                    printf("  N=%5d  B=%8d | SKIPPED (insufficient GPU memory)\n",
                           Ns[ni], Bs[bi]);
                    continue;
                }
                benchmark_fft(Ns[ni], Bs[bi], iters);
            }
        }
    }

    printf("\n  Benchmark complete.\n\n");
    return 0;
}
