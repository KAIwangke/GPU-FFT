#include "common.hpp"
// Stage 2: Memory coalescing and bank conflict optimization
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

__global__ void fft_butterfly_optimized(float* d_r, int width, int height, int m, bool is_row) {
    extern __shared__ float s_data[];
    
    const int idx = is_row ? blockIdx.x : blockIdx.y;
    const int tid = threadIdx.x;
    const int stride = is_row ? width : height;
    
    if (idx >= (is_row ? height : width)) return;
    
    // Coalesced loading into shared memory
    for (int i = tid; i < stride; i += blockDim.x) {
        const int global_idx = is_row ? 
            (idx * width + i) * 2 : 
            (i * width + idx) * 2;
            
        const int shared_idx = i * 2;
        s_data[shared_idx] = d_r[global_idx];
        s_data[shared_idx + 1] = d_r[global_idx + 1];
    }
    __syncthreads();
    
    // Butterfly computations
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
        
        s_data[shared_idx_a] = a_real;
        s_data[shared_idx_a + 1] = a_imag;
        s_data[shared_idx_b] = b_real;
        s_data[shared_idx_b + 1] = b_imag;
    }
    __syncthreads();
    
    // Coalesced storing back to global memory
    for (int i = tid; i < stride; i += blockDim.x) {
        const int global_idx = is_row ? 
            (idx * width + i) * 2 : 
            (i * width + idx) * 2;
            
        const int shared_idx = i * 2;
        d_r[global_idx] = s_data[shared_idx];
        d_r[global_idx + 1] = s_data[shared_idx + 1];
    }
}

void compute_2d_fft(float* d_data, int width, int height) {
    dim3 block(256);
    dim3 grid(height, 1); // For row-wise FFT
    
    // Calculate shared memory size
    size_t shared_mem_size = std::max(width, height) * 2 * sizeof(float);
    
    // Row-wise FFT
    for (int step = 2; step <= width; step <<= 1) {
        fft_butterfly_optimized<<<grid, block, shared_mem_size>>>(
            d_data, width, height, step, true);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // Reconfigure grid for column-wise FFT
    grid.x = width;
    grid.y = 1;
    
    // Column-wise FFT
    for (int step = 2; step <= height; step <<= 1) {
        fft_butterfly_optimized<<<grid, block, shared_mem_size>>>(
            d_data, width, height, step, false);
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

    // Copy data to device using pinned memory for better transfer performance
    float* h_pinned;
    CHECK_CUDA(cudaMallocHost(&h_pinned, size));
    std::memcpy(h_pinned, h_data.data(), size);
    CHECK_CUDA(cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice));

    // Perform FFT
    compute_2d_fft(d_data, width, height);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_pinned, d_data, size, cudaMemcpyDeviceToHost));
    std::memcpy(h_data.data(), h_pinned, size);

    // Stop timing
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Stage2: Total execution time: " << milliseconds << " ms" << std::endl;


    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFreeHost(h_pinned));
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}