#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

#define MATRIX_SIZE 64
#define BLOCK_SIZE 256
#define SHARED_MEM_SIZE 2048
#define M_PI 3.14159265358979323846

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void fft_butterfly_rows_optimized(float* d_r, int width, int height, int m) {
    __shared__ float s_data[SHARED_MEM_SIZE];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= height) return;
    
    // Load data into shared memory
    if (tid < width) {
        int base_idx = row * width * 2;
        s_data[tid*2] = d_r[base_idx + tid*2];
        s_data[tid*2 + 1] = d_r[base_idx + tid*2 + 1];
    }
    __syncthreads();
    
    if (tid < m/2) {
        float angle = (2.0f * M_PI * tid) / m;
        float t_real = cosf(angle);  // Changed from __cosf to cosf
        float t_imag = -sinf(angle); // Changed from __sinf to sinf
        
        int base_idx = row * width * 2;
        int k = tid;
        
        float u_real = s_data[k*2];
        float u_imag = s_data[k*2 + 1];
        float v_real = s_data[(k + m/2)*2];
        float v_imag = s_data[(k + m/2)*2 + 1];
        
        float tr_real = t_real * v_real - t_imag * v_imag;
        float tr_imag = t_real * v_imag + t_imag * v_real;
        
        d_r[base_idx + k*2] = u_real + tr_real;
        d_r[base_idx + k*2 + 1] = u_imag + tr_imag;
        d_r[base_idx + (k + m/2)*2] = u_real - tr_real;
        d_r[base_idx + (k + m/2)*2 + 1] = u_imag - tr_imag;
    }
}

void compute_2d_optimized(float* buffer, int rows, int cols, int sample_rate, const char* filename) {
    float *h_buffer = nullptr;
    float *d_buffer = nullptr;
    
    // Allocate host memory (changed from pinned to regular memory)
    h_buffer = new float[rows * cols * 2];
    
    // Initialize real and imaginary parts
    for (int i = 0; i < rows * cols; i++) {
        h_buffer[i*2] = buffer[i];      // Real part
        h_buffer[i*2 + 1] = 0.0f;       // Imaginary part
    }
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_buffer, rows * cols * 2 * sizeof(float)));
    
    // Start timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_buffer, h_buffer, rows * cols * 2 * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Process rows
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(rows);
    
    for (int i = 1; i <= 6; i++) {  // log2(64) = 6
        int m = 1 << i;
        fft_butterfly_rows_optimized<<<gridDim, blockDim>>>(d_buffer, cols, rows, m);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_buffer, d_buffer, rows * cols * 2 * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
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
    delete[] h_buffer;
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