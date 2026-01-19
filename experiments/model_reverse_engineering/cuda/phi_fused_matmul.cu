/**
 * Fused φ-Decode + MatMul CUDA Kernel
 * ====================================
 * 
 * Decodes φ-encoded weights on-the-fly during matrix multiplication.
 * This avoids the memory bandwidth cost of storing decoded floats.
 * 
 * Key optimization: Load compressed data, decode in registers, compute.
 * 
 * Compile:
 *   nvcc -O3 -arch=sm_86 -o phi_fused_matmul phi_fused_matmul.cu -lcudart
 * 
 * Author: TruthSpace LCM Team
 * License: GPLv3
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Constants
#define PHI 1.6180339887498949f
#define SCALE 1024
#define TILE_SIZE 16
#define LUT_SIZE 65536

// Global LUT (in device memory)
__device__ float d_phi_lut[LUT_SIZE];

// Decode helper
__device__ __forceinline__ float phi_decode(signed char sign, short exp) {
    return sign * d_phi_lut[exp + 32768];
}

/**
 * Fused φ-decode matmul with tiling.
 * 
 * C = decode(A) @ decode(B)
 * 
 * Where A and B are stored as (signs, exponents) pairs.
 */
__global__ void phi_fused_matmul_kernel(
    const signed char* __restrict__ A_signs,
    const short* __restrict__ A_exps,
    const signed char* __restrict__ B_signs,
    const short* __restrict__ B_exps,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load and decode A tile
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            int a_idx = row * K + a_col;
            As[ty][tx] = phi_decode(A_signs[a_idx], A_exps[a_idx]);
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load and decode B tile
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            int b_idx = b_row * N + col;
            Bs[ty][tx] = phi_decode(B_signs[b_idx], B_exps[b_idx]);
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Standard float32 matmul for comparison.
 */
__global__ void float_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Initialize LUT
void init_phi_lut() {
    float* h_lut = (float*)malloc(LUT_SIZE * sizeof(float));
    
    for (int i = 0; i < LUT_SIZE; i++) {
        int exp = i - 32768;
        h_lut[i] = powf(PHI, (float)exp / SCALE);
    }
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_phi_lut, h_lut, LUT_SIZE * sizeof(float)));
    free(h_lut);
}

int main() {
    printf("Fused φ-Decode MatMul Benchmark\n");
    printf("================================\n\n");
    
    init_phi_lut();
    printf("LUT initialized: %d entries, %.1f KB\n\n", LUT_SIZE, LUT_SIZE * sizeof(float) / 1024.0f);
    
    // Test sizes
    int sizes[][3] = {
        {512, 128, 512},
        {1024, 128, 1024},
        {2048, 128, 2048},
        {4096, 128, 4096},
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("%-20s %12s %12s %10s %12s\n", "Size", "Float32(ms)", "φ-Fused(ms)", "Speedup", "Compression");
    printf("------------------------------------------------------------------------\n");
    
    for (int s = 0; s < n_sizes; s++) {
        int M = sizes[s][0], K = sizes[s][1], N = sizes[s][2];
        
        // Allocate host memory
        float* h_A = (float*)malloc(M * K * sizeof(float));
        float* h_B = (float*)malloc(K * N * sizeof(float));
        signed char* h_A_signs = (signed char*)malloc(M * K);
        short* h_A_exps = (short*)malloc(M * K * sizeof(short));
        signed char* h_B_signs = (signed char*)malloc(K * N);
        short* h_B_exps = (short*)malloc(K * N * sizeof(short));
        
        // Initialize
        srand(42);
        for (int i = 0; i < M * K; i++) {
            float val = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
            h_A[i] = val;
            h_A_signs[i] = val >= 0 ? 1 : -1;
            h_A_exps[i] = (short)(logf(fabsf(val) + 1e-10f) / logf(PHI) * SCALE);
        }
        for (int i = 0; i < K * N; i++) {
            float val = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
            h_B[i] = val;
            h_B_signs[i] = val >= 0 ? 1 : -1;
            h_B_exps[i] = (short)(logf(fabsf(val) + 1e-10f) / logf(PHI) * SCALE);
        }
        
        // Allocate device memory
        float *d_A, *d_B, *d_C_float, *d_C_phi;
        signed char *d_A_signs, *d_B_signs;
        short *d_A_exps, *d_B_exps;
        
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_float, M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_phi, M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A_signs, M * K));
        CUDA_CHECK(cudaMalloc(&d_A_exps, M * K * sizeof(short)));
        CUDA_CHECK(cudaMalloc(&d_B_signs, K * N));
        CUDA_CHECK(cudaMalloc(&d_B_exps, K * N * sizeof(short)));
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_A_signs, h_A_signs, M * K, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_A_exps, h_A_exps, M * K * sizeof(short), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_signs, h_B_signs, K * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_exps, h_B_exps, K * N * sizeof(short), cudaMemcpyHostToDevice));
        
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        
        // Warmup
        float_matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C_float, M, K, N);
        phi_fused_matmul_kernel<<<blocks, threads>>>(d_A_signs, d_A_exps, d_B_signs, d_B_exps, d_C_phi, M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        int iterations = 100;
        
        // Float32
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            float_matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C_float, M, K, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_float;
        CUDA_CHECK(cudaEventElapsedTime(&ms_float, start, stop));
        ms_float /= iterations;
        
        // φ-Fused
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            phi_fused_matmul_kernel<<<blocks, threads>>>(d_A_signs, d_A_exps, d_B_signs, d_B_exps, d_C_phi, M, K, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_phi;
        CUDA_CHECK(cudaEventElapsedTime(&ms_phi, start, stop));
        ms_phi /= iterations;
        
        float speedup = ms_float / ms_phi;
        
        // Memory comparison
        size_t float_bytes = (M * K + K * N) * sizeof(float);
        size_t phi_bytes = (M * K + K * N) * 3;  // 1 sign + 2 exp
        float compression = (float)float_bytes / phi_bytes;
        
        char size_str[32];
        sprintf(size_str, "%dx%dx%d", M, K, N);
        printf("%-20s %12.3f %12.3f %9.2fx %11.2fx\n", size_str, ms_float, ms_phi, speedup, compression);
        
        // Cleanup
        free(h_A); free(h_B);
        free(h_A_signs); free(h_A_exps);
        free(h_B_signs); free(h_B_exps);
        cudaFree(d_A); cudaFree(d_B);
        cudaFree(d_C_float); cudaFree(d_C_phi);
        cudaFree(d_A_signs); cudaFree(d_A_exps);
        cudaFree(d_B_signs); cudaFree(d_B_exps);
    }
    
    printf("\nNote: φ-Fused loads 1.33x less data from global memory.\n");
    printf("The speedup comes from reduced memory bandwidth pressure.\n");
    
    return 0;
}
