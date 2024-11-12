#include "common.hpp"

__global__ void fft_1d_row(float* d_r, int width, int height, int step) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    int half_step = step / 2;
    
    // Add bounds checking
    if (row >= height || col >= half_step) return;
    
    int idx1 = row * width * 2 + col * 2;
    int idx2 = idx1 + step * 2;
    
    // Add bounds checking for memory access
    if (idx2 + 1 >= width * height * 2) return;
    
    float angle = -2.0f * M_PI * col / step;
    float t_real = cosf(angle);
    float t_imag = sinf(angle);
    
    float u_real = d_r[idx1];
    float u_imag = d_r[idx1 + 1];
    float v_real = d_r[idx2];
    float v_imag = d_r[idx2 + 1];
    
    float tr = t_real * v_real - t_imag * v_imag;
    float ti = t_real * v_imag + t_imag * v_real;
    
    d_r[idx1] = u_real + tr;
    d_r[idx1 + 1] = u_imag + ti;
    d_r[idx2] = u_real - tr;
    d_r[idx2 + 1] = u_imag - ti;
}

__global__ void fft_1d_column(float* d_r, int width, int height, int step) {
    int col = blockIdx.x;
    int row = threadIdx.x;
    int half_step = step / 2;
    
    // Add bounds checking
    if (col >= width || row >= half_step) return;
    
    int idx1 = col * 2 + row * width * 2;
    int idx2 = idx1 + step * width * 2;
    
    // Add bounds checking for memory access
    if (idx2 + 1 >= width * height * 2) return;
    
    float angle = -2.0f * M_PI * row / step;
    float t_real = cosf(angle);
    float t_imag = sinf(angle);
    
    float u_real = d_r[idx1];
    float u_imag = d_r[idx1 + 1];
    float v_real = d_r[idx2];
    float v_imag = d_r[idx2 + 1];
    
    float tr = t_real * v_real - t_imag * v_imag;
    float ti = t_real * v_imag + t_imag * v_real;
    
    d_r[idx1] = u_real + tr;
    d_r[idx1 + 1] = u_imag + ti;
    d_r[idx2] = u_real - tr;
    d_r[idx2 + 1] = u_imag - ti;
}

void compute_2d_fft(float* d_data, int width, int height) {
    // Calculate optimal block size
    int max_threads = 256;
    int block_size = std::min(max_threads, nextPowerOf2(width/2));
    
    dim3 block(block_size);
    dim3 grid_row((height + block.x - 1) / block.x);
    dim3 grid_col((width + block.x - 1) / block.x);
    
    // Row-wise FFT
    for (int step = 2; step <= width; step <<= 1) {
        fft_1d_row<<<grid_row, block>>>(d_data, width, height, step);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // Column-wise FFT
    for (int step = 2; step <= height; step <<= 1) {
        fft_1d_column<<<grid_col, block>>>(d_data, width, height, step);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

// Helper function to get next power of 2
inline int nextPowerOf2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
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

    // Create CUDA events for timing
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

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Stage0: Total execution time: " << milliseconds << " ms" << std::endl;


    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}