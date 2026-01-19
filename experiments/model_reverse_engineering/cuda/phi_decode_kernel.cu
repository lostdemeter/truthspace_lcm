/**
 * φ-Decode CUDA Kernel
 * ====================
 * 
 * On-the-fly decoding of φ-encoded weights for memory bandwidth optimization.
 * 
 * Storage format:
 *   - sign: int8 (-1, 0, or 1)
 *   - exponent: int16 (scaled by 8192)
 *   - value = sign × φ^(exponent/8192)
 * 
 * Memory bandwidth win:
 *   - Float32: 32 bits per value
 *   - φ-encoded: 24 bits per value (8 sign + 16 exp)
 *   - Compression: 1.33x (could be 1.9x with packed 17-bit)
 * 
 * Compile:
 *   nvcc -O3 -arch=sm_86 -o phi_decode_kernel phi_decode_kernel.cu -lcudart
 * 
 * Author: TruthSpace LCM Team
 * License: GPLv3
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
#define LOG_PHI 0.4812118250596034f
#define SCALE 8192
#define LUT_SIZE 65536  // Cover full int16 range

// LUT in constant memory (64KB limit, we use ~256KB so use global)
// For exponents in range [-32768, 32767], precompute φ^(exp/SCALE)
__device__ float d_phi_lut[LUT_SIZE];

// Kernel: Decode φ-encoded matrix to float
__global__ void phi_decode_kernel(
    const int8_t* __restrict__ signs,      // (M, N)
    const int16_t* __restrict__ exponents, // (M, N)
    float* __restrict__ output,            // (M, N)
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    
    if (idx < total) {
        int sign = signs[idx];
        int exp = exponents[idx];
        
        // LUT lookup (exp is int16, offset by 32768 to get positive index)
        int lut_idx = exp + 32768;
        float magnitude = d_phi_lut[lut_idx];
        
        output[idx] = sign * magnitude;
    }
}

// Kernel: Decode and immediately do matmul (fused for bandwidth)
// C = A_decoded @ B_decoded
// This is the key optimization: decode on-the-fly during matmul
__global__ void phi_decode_matmul_kernel(
    const int8_t* __restrict__ A_signs,      // (M, K)
    const int16_t* __restrict__ A_exponents,
    const int8_t* __restrict__ B_signs,      // (K, N)
    const int16_t* __restrict__ B_exponents,
    float* __restrict__ C,                   // (M, N)
    int M, int K, int N
) {
    // Each thread computes one element of C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        for (int k = 0; k < K; k++) {
            // Decode A[row, k]
            int a_idx = row * K + k;
            int a_sign = A_signs[a_idx];
            int a_exp = A_exponents[a_idx];
            float a_val = a_sign * d_phi_lut[a_exp + 32768];
            
            // Decode B[k, col]
            int b_idx = k * N + col;
            int b_sign = B_signs[b_idx];
            int b_exp = B_exponents[b_idx];
            float b_val = b_sign * d_phi_lut[b_exp + 32768];
            
            sum += a_val * b_val;
        }
        
        C[row * N + col] = sum;
    }
}

// Tiled version for better memory coalescing
#define TILE_SIZE 16

__global__ void phi_decode_matmul_tiled_kernel(
    const int8_t* __restrict__ A_signs,
    const int16_t* __restrict__ A_exponents,
    const int8_t* __restrict__ B_signs,
    const int16_t* __restrict__ B_exponents,
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
        // Load A tile (decode on load)
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            int a_idx = row * K + a_col;
            int a_sign = A_signs[a_idx];
            int a_exp = A_exponents[a_idx];
            As[ty][tx] = a_sign * d_phi_lut[a_exp + 32768];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load B tile (decode on load)
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            int b_idx = b_row * N + col;
            int b_sign = B_signs[b_idx];
            int b_exp = B_exponents[b_idx];
            Bs[ty][tx] = b_sign * d_phi_lut[b_exp + 32768];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to initialize LUT
void init_phi_lut() {
    float* h_lut = (float*)malloc(LUT_SIZE * sizeof(float));
    
    for (int i = 0; i < LUT_SIZE; i++) {
        int exp = i - 32768;  // Convert to signed
        h_lut[i] = powf(PHI, (float)exp / SCALE);
    }
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_phi_lut, h_lut, LUT_SIZE * sizeof(float)));
    free(h_lut);
    
    printf("φ-LUT initialized: %d entries, %.1f KB\n", LUT_SIZE, LUT_SIZE * sizeof(float) / 1024.0f);
}

// Wrapper for Python (via ctypes or pybind11)
extern "C" {

void phi_init_lut() {
    init_phi_lut();
}

void phi_decode(
    const int8_t* d_signs,
    const int16_t* d_exponents,
    float* d_output,
    int M, int N
) {
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    phi_decode_kernel<<<blocks, threads>>>(d_signs, d_exponents, d_output, M, N);
    CUDA_CHECK(cudaGetLastError());
}

void phi_decode_matmul(
    const int8_t* d_A_signs,
    const int16_t* d_A_exponents,
    const int8_t* d_B_signs,
    const int16_t* d_B_exponents,
    float* d_C,
    int M, int K, int N,
    int use_tiled
) {
    if (use_tiled) {
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        phi_decode_matmul_tiled_kernel<<<blocks, threads>>>(
            d_A_signs, d_A_exponents, d_B_signs, d_B_exponents, d_C, M, K, N
        );
    } else {
        dim3 threads(16, 16);
        dim3 blocks((N + 15) / 16, (M + 15) / 16);
        phi_decode_matmul_kernel<<<blocks, threads>>>(
            d_A_signs, d_A_exponents, d_B_signs, d_B_exponents, d_C, M, K, N
        );
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"

// Test program
int main(int argc, char** argv) {
    printf("φ-Decode CUDA Kernel Test\n");
    printf("=========================\n\n");
    
    // Initialize LUT
    init_phi_lut();
    
    // Test dimensions
    int M = 1024, K = 128, N = 1024;
    printf("Test: (%d x %d) @ (%d x %d) = (%d x %d)\n", M, K, K, N, M, N);
    
    // Allocate host memory
    int8_t* h_A_signs = (int8_t*)malloc(M * K * sizeof(int8_t));
    int16_t* h_A_exponents = (int16_t*)malloc(M * K * sizeof(int16_t));
    int8_t* h_B_signs = (int8_t*)malloc(K * N * sizeof(int8_t));
    int16_t* h_B_exponents = (int16_t*)malloc(K * N * sizeof(int16_t));
    float* h_C = (float*)malloc(M * N * sizeof(float));
    float* h_C_ref = (float*)malloc(M * N * sizeof(float));
    
    // Initialize with random φ-encoded values
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A_signs[i] = (rand() % 2) * 2 - 1;  // -1 or 1
        h_A_exponents[i] = (rand() % 20000) - 10000;  // Range [-10000, 10000]
    }
    for (int i = 0; i < K * N; i++) {
        h_B_signs[i] = (rand() % 2) * 2 - 1;
        h_B_exponents[i] = (rand() % 20000) - 10000;
    }
    
    // Compute reference on CPU
    printf("Computing CPU reference...\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int a_idx = i * K + k;
                int b_idx = k * N + j;
                float a_val = h_A_signs[a_idx] * powf(PHI, (float)h_A_exponents[a_idx] / SCALE);
                float b_val = h_B_signs[b_idx] * powf(PHI, (float)h_B_exponents[b_idx] / SCALE);
                sum += a_val * b_val;
            }
            h_C_ref[i * N + j] = sum;
        }
    }
    
    // Allocate device memory
    int8_t *d_A_signs, *d_B_signs;
    int16_t *d_A_exponents, *d_B_exponents;
    float *d_C;
    
    CUDA_CHECK(cudaMalloc(&d_A_signs, M * K * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_A_exponents, M * K * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_B_signs, K * N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_B_exponents, K * N * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A_signs, h_A_signs, M * K * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_exponents, h_A_exponents, M * K * sizeof(int16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_signs, h_B_signs, K * N * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_exponents, h_B_exponents, K * N * sizeof(int16_t), cudaMemcpyHostToDevice));
    
    // Warm up
    phi_decode_matmul(d_A_signs, d_A_exponents, d_B_signs, d_B_exponents, d_C, M, K, N, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    int iterations = 100;
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        phi_decode_matmul(d_A_signs, d_A_exponents, d_B_signs, d_B_exponents, d_C, M, K, N, 1);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabsf(h_C[i] - h_C_ref[i]);
        if (error > max_error) max_error = error;
    }
    
    // Memory bandwidth calculation
    // Input: M*K*3 + K*N*3 bytes (sign + 2 bytes exp)
    // Output: M*N*4 bytes
    size_t input_bytes = (M * K + K * N) * 3;
    size_t output_bytes = M * N * 4;
    float bandwidth_gb = (input_bytes + output_bytes) / (avg_ms / 1000.0f) / 1e9;
    
    // Compare to float32 input
    size_t float_input_bytes = (M * K + K * N) * 4;
    float compression = (float)float_input_bytes / input_bytes;
    
    printf("\nResults:\n");
    printf("  Max error: %.6e\n", max_error);
    printf("  Avg time: %.3f ms\n", avg_ms);
    printf("  Bandwidth: %.1f GB/s\n", bandwidth_gb);
    printf("  Compression vs float32: %.2fx\n", compression);
    printf("  φ-encoded input: %.1f KB\n", input_bytes / 1024.0f);
    printf("  Float32 input would be: %.1f KB\n", float_input_bytes / 1024.0f);
    
    // Cleanup
    free(h_A_signs); free(h_A_exponents);
    free(h_B_signs); free(h_B_exponents);
    free(h_C); free(h_C_ref);
    cudaFree(d_A_signs); cudaFree(d_A_exponents);
    cudaFree(d_B_signs); cudaFree(d_B_exponents);
    cudaFree(d_C);
    
    return 0;
}
