// Constants and helper functions
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024
#define USE_TEXTURE_MEMORY 1

// Precomputed twiddle factors texture
texture<float2, 1, cudaReadModeElementType> twiddle_tex;

// Helper function to compute twiddle factors
__host__ void precompute_twiddle_factors(int max_size) {
    float2* h_twiddle = new float2[max_size/2];
    for (int i = 0; i < max_size/2; i++) {
        float angle = -2.0f * M_PI * i / max_size;
        h_twiddle[i].x = cosf(angle);
        h_twiddle[i].y = sinf(angle);
    }
    
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
    cudaArray* d_twiddle_array;
    cudaMallocArray(&d_twiddle_array, &desc, max_size/2, 1);
    cudaMemcpyToArray(d_twiddle_array, 0, 0, h_twiddle, sizeof(float2)*max_size/2, cudaMemcpyHostToDevice);
    
    twiddle_tex.addressMode[0] = cudaAddressModeWrap;
    twiddle_tex.filterMode = cudaFilterModeLinear;
    twiddle_tex.normalized = false;
    cudaBindTextureToArray(twiddle_tex, d_twiddle_array);
    
    delete[] h_twiddle;
}

// Optimized butterfly computation kernel
__device__ inline void butterfly_operation(float2& a, float2& b, float2 twiddle) {
    float2 temp;
    temp.x = a.x + (twiddle.x * b.x - twiddle.y * b.y);
    temp.y = a.y + (twiddle.x * b.y + twiddle.y * b.x);
    b.x = a.x - (twiddle.x * b.x - twiddle.y * b.y);
    b.y = a.y - (twiddle.x * b.y + twiddle.y * b.x);
    a = temp;
}

// Optimized row-wise FFT kernel using warp-level optimization
__global__ void fft_butterfly_rows_optimized_v2(float2* d_data, int width, int height, int stage, int butterfly_distance) {
    __shared__ float2 s_data[MAX_BLOCK_SIZE];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row = bid;
    
    if (row >= height) return;
    
    // Load data into shared memory with vector loads
    float2* row_data = &d_data[row * width];
    if (tid < width) {
        s_data[tid] = row_data[tid];
    }
    __syncthreads();
    
    // Butterfly computations within warps
    const int lane_id = tid & (WARP_SIZE-1);
    const int warp_id = tid >> 5;
    
    #pragma unroll
    for (int j = 0; j < stage; j++) {
        int group_size = 1 << j;
        int pair_distance = group_size;
        
        int group_thread_id = tid & (group_size - 1);
        int base_idx = (tid - group_thread_id) << 1;
        
        float2 twiddle;
        if (USE_TEXTURE_MEMORY) {
            twiddle = tex1Dfetch(twiddle_tex, group_thread_id);
        } else {
            float angle = -2.0f * M_PI * group_thread_id / (group_size * 2);
            twiddle.x = __cosf(angle);
            twiddle.y = __sinf(angle);
        }
        
        butterfly_operation(s_data[base_idx + group_thread_id],
                          s_data[base_idx + group_thread_id + pair_distance],
                          twiddle);
        
        __syncthreads();
    }
    
    // Store results back to global memory
    if (tid < width) {
        row_data[tid] = s_data[tid];
    }
}

// Optimized column-wise FFT kernel with bank conflict avoidance
__global__ void fft_butterfly_columns_optimized_v2(float2* d_data, int width, int height, int stage, int butterfly_distance) {
    __shared__ float2 s_data[MAX_BLOCK_SIZE + 1];  // +1 to avoid bank conflicts
    
    const int tid = threadIdx.x;
    const int col = blockIdx.x;
    
    if (col >= width) return;
    
    // Load column data with stride to avoid bank conflicts
    for (int i = tid; i < height; i += blockDim.x) {
        s_data[i] = d_data[i * width + col];
    }
    __syncthreads();
    
    // Perform butterfly operations on columns
    for (int i = tid; i < height/2; i += blockDim.x) {
        float2 twiddle;
        if (USE_TEXTURE_MEMORY) {
            twiddle = tex1Dfetch(twiddle_tex, i);
        } else {
            float angle = -2.0f * M_PI * i / height;
            twiddle.x = __cosf(angle);
            twiddle.y = __sinf(angle);
        }
        
        butterfly_operation(s_data[i],
                          s_data[i + height/2],
                          twiddle);
    }
    __syncthreads();
    
    // Store results back
    for (int i = tid; i < height; i += blockDim.x) {
        d_data[i * width + col] = s_data[i];
    }
}

// Main FFT function with optimization
void fft_2d_optimized(float2* d_data, int width, int height) {
    dim3 block_rows(MAX_BLOCK_SIZE);
    dim3 grid_rows((height + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE);
    
    dim3 block_cols(MAX_BLOCK_SIZE);
    dim3 grid_cols((width + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE);
    
    // Row-wise FFT
    for (int stage = 1; stage <= log2(width); stage++) {
        int butterfly_distance = 1 << (stage - 1);
        fft_butterfly_rows_optimized_v2<<<grid_rows, block_rows>>>(
            d_data, width, height, stage, butterfly_distance);
    }
    
    // Column-wise FFT
    for (int stage = 1; stage <= log2(height); stage++) {
        int butterfly_distance = 1 << (stage - 1);
        fft_butterfly_columns_optimized_v2<<<grid_cols, block_cols>>>(
            d_data, width, height, stage, butterfly_distance);
    }
}