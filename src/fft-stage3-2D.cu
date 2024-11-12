#include "common.hpp"
// Stage 3: Warp-level optimization
__device__ inline void warp_butterfly(volatile float* s_real, volatile float* s_imag, 
                                    int tid, int stride) {
    float angle = -2.0f * M_PI * (tid % stride) / (stride * 2);
    float t_real = cosf(angle);
    float t_imag = sinf(angle);
    
    int pos = 2 * tid - (tid & (stride - 1));
    float u_real = s_real[pos];
    float u_imag = s_imag[pos];
    float v_real = s_real[pos + stride];
    float v_imag = s_imag[pos + stride];
    
    float tr = t_real * v_real - t_imag * v_imag;
    float ti = t_real * v_imag + t_imag * v_real;
    
    s_real[pos] = u_real + tr;
    s_imag[pos] = u_imag + ti;
    s_real[pos + stride] = u_real - tr;
    s_imag[pos + stride] = u_imag - ti;
}

__global__ void fft_warp_optimized(float* d_r, int width, int height, int step, bool is_row) {
    extern __shared__ float s_data[];
    volatile float* s_real = s_data;
    volatile float* s_imag = s_data + blockDim.x;
    
    const int idx = is_row ? blockIdx.x : blockIdx.y;
    const int tid = threadIdx.x;
    const int stride = is_row ? width : height;
    
    if (idx >= (is_row ? height : width)) return;
    
    // Load data
    for (int i = tid; i < stride; i += blockDim.x) {
        const int global_idx = is_row ? 
            (idx * width + i) * 2 : 
            (i * width + idx) * 2;
        s_real[i] = d_r[global_idx];
        s_imag[i] = d_r[global_idx + 1];
    }
    __syncthreads();
    
    // Warp-level FFT
    for (int s = 1; s <= 32 && s <= stride/2; s *= 2) {
        if (tid < stride/2) {
            warp_butterfly(s_real, s_imag, tid, s);
        }
        __syncwarp();
    }
    
    // Block-level FFT
    for (int s = 64; s <= stride; s *= 2) {
        for (int i = tid; i < stride/2; i += blockDim.x) {
            warp_butterfly(s_real, s_imag, i, s);
        }
        __syncthreads();
    }
    
    // Store results
    for (int i = tid; i < stride; i += blockDim.x) {
        const int global_idx = is_row ? 
            (idx * width + i) * 2 : 
            (i * width + idx) * 2;
        d_r[global_idx] = s_real[i];
        d_r[global_idx + 1] = s_imag[i];
    }
}

void compute_2d_fft(float* d_data, int width, int height) {
    // Use warp size for block dimension to optimize warp-level operations
    dim3 block(32);  // CUDA warp size
    dim3 grid(height, 1);  // For row-wise FFT
    
    // Calculate shared memory size - need double the space for separate real/imaginary arrays
    size_t shared_mem_size = 2 * std::max(width, height) * sizeof(float);
    
    // Row-wise FFT
    for (int step = 2; step <= width; step <<= 1) {
        fft_warp_optimized<<<grid, block, shared_mem_size>>>(
            d_data, width, height, step, true);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // Reconfigure grid for column-wise FFT
    grid.x = width;
    grid.y = 1;
    
    // Column-wise FFT
    for (int step = 2; step <= height; step <<= 1) {
        fft_warp_optimized<<<grid, block, shared_mem_size>>>(
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

    // Check if dimensions are compatible with warp-level operations
    if (width % 32 != 0 || height % 32 != 0) {
        std::cout << "Warning: Matrix dimensions not multiples of warp size (32). "
                  << "Performance may be suboptimal." << std::endl;
    }

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

    // Use pinned memory for better transfer performance
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
    std::cout << "Stage3: Total execution time: " << milliseconds << " ms" << std::endl;

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFreeHost(h_pinned));
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}