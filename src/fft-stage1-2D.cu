#include "common.hpp"

// Stage 1: Shared memory optimized kernel
__global__ void fft_butterfly_rows_optimized(float* d_r, int width, int height, int m) {
    extern __shared__ float s_data[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= height) return;
    
    // Load data into shared memory
    for (int i = tid; i < width; i += blockDim.x) {
        int base_idx = row * width * 2;
        s_data[i*2] = d_r[base_idx + i*2];
        s_data[i*2 + 1] = d_r[base_idx + i*2 + 1];
    }
    __syncthreads();
    
    for (int i = tid; i < m/2; i += blockDim.x) {
        float angle = (2.0f * M_PI * i) / m;
        float t_real = cosf(angle);
        float t_imag = -sinf(angle);
        
        int k = i;
        float u_real = s_data[k*2];
        float u_imag = s_data[k*2 + 1];
        float v_real = s_data[(k + m/2)*2];
        float v_imag = s_data[(k + m/2)*2 + 1];
        
        float tr_real = t_real * v_real - t_imag * v_imag;
        float tr_imag = t_real * v_imag + t_imag * v_real;
        
        s_data[k*2] = u_real + tr_real;
        s_data[k*2 + 1] = u_imag + tr_imag;
        s_data[(k + m/2)*2] = u_real - tr_real;
        s_data[(k + m/2)*2 + 1] = u_imag - tr_imag;
    }
    __syncthreads();
    
    // Store results back to global memory
    for (int i = tid; i < width; i += blockDim.x) {
        int base_idx = row * width * 2;
        d_r[base_idx + i*2] = s_data[i*2];
        d_r[base_idx + i*2 + 1] = s_data[i*2 + 1];
    }
}

void compute_2d_fft(float* d_data, int width, int height) {
    dim3 block(256);
    dim3 grid_row((height + block.x - 1) / block.x);
    
    // Calculate shared memory size - needs to accommodate the largest row/column
    size_t shared_mem_size = width * 2 * sizeof(float);  // *2 for complex numbers
    
    // Row-wise FFT
    for (int step = 2; step <= width; step <<= 1) {
        fft_butterfly_rows_optimized<<<grid_row, block, shared_mem_size>>>(
            d_data, width, height, step);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // Column-wise FFT - we'll reuse the same kernel with transposed indices
    dim3 grid_col((width + block.x - 1) / block.x);
    for (int step = 2; step <= height; step <<= 1) {
        fft_butterfly_rows_optimized<<<grid_col, block, shared_mem_size>>>(
            d_data, height, width, step);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    // Get matrix dimensions
    int width, height;
    get_matrix_dimensions(argv[1], width, height);
    std::cout << "Processing matrix of size " << width << "x" << height << std::endl;

    // Read input data
    std::vector<float> h_data;
    if (!read_matrix_data(argv[1], h_data, width, height)) {
        return 1;
    }

    // Allocate device memory
    float* d_data;
    size_t size = width * height * 2 * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_data, size));

    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Start timing
    CHECK_CUDA(cudaEventRecord(start));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice));

    // Perform FFT
    compute_2d_fft(d_data, width, height);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, size, cudaMemcpyDeviceToHost));

    // Stop timing
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Total execution time: " << milliseconds << " ms" << std::endl;

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}