#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

#define MATRIX_SIZE 2000  // Changed from 64 to 256
#define BLOCK_SIZE 256   // Keep block size at 256
#define SHARED_MEM_SIZE 4096  // Increased from 2048 to 4096 for larger matrix
#define M_PI 3.14159265358979323846

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Optimized butterfly computation
__device__ inline void butterfly_operation(float& a_real, float& a_imag, 
                                         float& b_real, float& b_imag,
                                         float t_real, float t_imag) {
    float temp_real = a_real;
    float temp_imag = a_imag;
    
    float tr_real = t_real * b_real - t_imag * b_imag;
    float tr_imag = t_real * b_imag + t_imag * b_real;
    
    a_real = temp_real + tr_real;
    a_imag = temp_imag + tr_imag;
    b_real = temp_real - tr_real;
    b_imag = temp_imag - tr_imag;
}

__global__ void fft_butterfly_rows_optimized(float* d_r, int width, int height, int m) {
    extern __shared__ float s_data[];
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= height) return;
    
    // Coalesced load into shared memory using multiple loads per thread if needed
    for (int i = tid; i < width; i += blockDim.x) {
        const int base_idx = row * width * 2;
        const int shared_idx = i * 2;
        s_data[shared_idx] = d_r[base_idx + shared_idx];
        s_data[shared_idx + 1] = d_r[base_idx + shared_idx + 1];
    }
    __syncthreads();
    
    // Butterfly computations with strided processing
    for (int i = tid; i < m/2; i += blockDim.x) {
        const float angle = (2.0f * M_PI * i) / m;
        const float t_real = cosf(angle);
        const float t_imag = -sinf(angle);
        
        const int k = i;
        const int shared_idx_a = k * 2;
        const int shared_idx_b = (k + m/2) * 2;
        
        // Load data
        float a_real = s_data[shared_idx_a];
        float a_imag = s_data[shared_idx_a + 1];
        float b_real = s_data[shared_idx_b];
        float b_imag = s_data[shared_idx_b + 1];
        
        // Perform butterfly operation
        butterfly_operation(a_real, a_imag, b_real, b_imag, t_real, t_imag);
        
        // Store results back to shared memory
        s_data[shared_idx_a] = a_real;
        s_data[shared_idx_a + 1] = a_imag;
        s_data[shared_idx_b] = b_real;
        s_data[shared_idx_b + 1] = b_imag;
    }
    __syncthreads();
    
    // Store results back to global memory
    for (int i = tid; i < width; i += blockDim.x) {
        const int base_idx = row * width * 2;
        const int shared_idx = i * 2;
        d_r[base_idx + shared_idx] = s_data[shared_idx];
        d_r[base_idx + shared_idx + 1] = s_data[shared_idx + 1];
    }
}

__global__ void fft_butterfly_cols_optimized(float* d_r, int width, int height, int m) {
    extern __shared__ float s_data[];
    
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (col >= width) return;
    
    // Load column data into shared memory with strided access
    for (int i = tid; i < height; i += blockDim.x) {
        const int global_idx = (i * width + col) * 2;
        const int shared_idx = i * 2;
        s_data[shared_idx] = d_r[global_idx];
        s_data[shared_idx + 1] = d_r[global_idx + 1];
    }
    __syncthreads();
    
    // Process column data with strided processing
    for (int i = tid; i < m/2; i += blockDim.x) {
        const float angle = (2.0f * M_PI * i) / m;
        const float t_real = cosf(angle);
        const float t_imag = -sinf(angle);
        
        const int shared_idx_a = i * 2;
        const int shared_idx_b = (i + m/2) * 2;
        
        float a_real = s_data[shared_idx_a];
        float a_imag = s_data[shared_idx_a + 1];
        float b_real = s_data[shared_idx_b];
        float b_imag = s_data[shared_idx_b + 1];
        
        butterfly_operation(a_real, a_imag, b_real, b_imag, t_real, t_imag);
        
        // Store results back to shared memory
        s_data[shared_idx_a] = a_real;
        s_data[shared_idx_a + 1] = a_imag;
        s_data[shared_idx_b] = b_real;
        s_data[shared_idx_b + 1] = b_imag;
    }
    __syncthreads();
    
    // Write results back to global memory
    for (int i = tid; i < height; i += blockDim.x) {
        const int global_idx = (i * width + col) * 2;
        const int shared_idx = i * 2;
        d_r[global_idx] = s_data[shared_idx];
        d_r[global_idx + 1] = s_data[shared_idx + 1];
    }
}

void compute_2d_optimized(float* buffer, int rows, int cols, int sample_rate, const char* filename) {
    float *h_buffer = nullptr;
    float *d_buffer = nullptr;
    
    size_t buffer_size = rows * cols * 2 * sizeof(float);
    
    // Allocate host memory using page-locked memory for better transfer performance
    CHECK_CUDA(cudaMallocHost(&h_buffer, buffer_size));
    
    // Initialize real and imaginary parts
    for (int i = 0; i < rows * cols; i++) {
        h_buffer[i*2] = buffer[i];
        h_buffer[i*2 + 1] = 0.0f;
    }
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_buffer, buffer_size));
    
    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_buffer, h_buffer, buffer_size, cudaMemcpyHostToDevice));
    
    // Process rows
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim_rows((rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 gridDim_cols((cols + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Row-wise FFT
    for (int i = 1; i <= (int)log2(cols); i++) {
        int m = 1 << i;
        fft_butterfly_rows_optimized<<<gridDim_rows, blockDim, SHARED_MEM_SIZE * sizeof(float)>>>
            (d_buffer, cols, rows, m);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // Column-wise FFT
    for (int i = 1; i <= (int)log2(rows); i++) {
        int m = 1 << i;
        fft_butterfly_cols_optimized<<<gridDim_cols, blockDim, SHARED_MEM_SIZE * sizeof(float)>>>
            (d_buffer, cols, rows, m);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    cout << "Elapsed time: " << milliseconds / 1000.0 << " seconds" << endl;
    
    // Save results
    ofstream outfile;
    outfile.open(string(filename) + ".out");
    outfile.precision(6);
    outfile << "2D FFT Output (first 5x5):" << endl;
    for (int i = 0; i < min(5, rows); i++) {
        for (int j = 0; j < min(5, cols); j++) {
            float real = h_buffer[(i * cols + j) * 2];
            float imag = h_buffer[(i * cols + j) * 2 + 1];
            outfile << "X[" << i << "][" << j << "] = " 
                   << real << " + " << imag << "i\t";
        }
        outfile << endl;
    }
    outfile.close();
    
    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUDA(cudaFreeHost(h_buffer));  // Free page-locked memory
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " [input_file]" << endl;
        return 1;
    }
    
    // Read input file
    ifstream infile(argv[1]);
    if (!infile) {
        cerr << "Error opening input file: " << argv[1] << endl;
        return 1;
    }
    
    vector<float> buffer;
    float value;
    while (infile >> value) {
        buffer.push_back(value);
    }
    infile.close();
    
    if (buffer.size() != MATRIX_SIZE * MATRIX_SIZE) {
        cerr << "Invalid input size. Expected " << MATRIX_SIZE * MATRIX_SIZE 
             << " elements, got " << buffer.size() << endl;
        return 1;
    }
    
    compute_2d_optimized(buffer.data(), MATRIX_SIZE, MATRIX_SIZE, 0, argv[1]);
    
    return 0;
}