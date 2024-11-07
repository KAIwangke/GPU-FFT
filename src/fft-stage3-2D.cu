#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cufft.h>
#include <nvToolsExt.h>

// Constants for multi-GPU management
#define MAX_GPUS 8
#define PROFILING_ENABLED 1
#define ERROR_CHECK 1

// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Structure for GPU context
struct GPUContext {
    int device_id;
    cudaStream_t stream;
    cufftHandle plan;
    float2* d_data;
    int chunk_size;
};

// Performance monitoring structure
struct PerfMetrics {
    float transfer_time;
    float compute_time;
    float total_time;
    float throughput;
};

class FFT2DMultiGPU {
private:
    GPUContext* contexts;
    int num_gpus;
    int width;
    int height;
    PerfMetrics metrics;
    
    // Advanced memory management
    cudaMemPool_t mem_pool[MAX_GPUS];
    
public:
    FFT2DMultiGPU(int w, int h) : width(w), height(h) {
        // Initialize available GPUs
        CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
        num_gpus = min(num_gpus, MAX_GPUS);
        contexts = new GPUContext[num_gpus];
        
        // Initialize each GPU
        for (int i = 0; i < num_gpus; i++) {
            initializeGPU(i);
        }
    }
    
    ~FFT2DMultiGPU() {
        cleanup();
    }
    
private:
    void initializeGPU(int gpu_index) {
        GPUContext& ctx = contexts[gpu_index];
        ctx.device_id = gpu_index;
        
        // Set device and create stream
        CHECK_CUDA(cudaSetDevice(gpu_index));
        CHECK_CUDA(cudaStreamCreate(&ctx.stream));
        
        // Initialize memory pool
        cudaMemPoolProps poolProps = {};
        poolProps.allocType = cudaMemAllocationTypePinned;
        poolProps.location.id = gpu_index;
        poolProps.location.type = cudaMemLocationTypeDevice;
        CHECK_CUDA(cudaMemPoolCreate(&mem_pool[gpu_index], &poolProps));
        
        // Calculate chunk size for this GPU
        ctx.chunk_size = (height / num_gpus) + (gpu_index < (height % num_gpus) ? 1 : 0);
        
        // Allocate device memory from pool
        size_t size = ctx.chunk_size * width * sizeof(float2);
        CHECK_CUDA(cudaMallocFromPool((void**)&ctx.d_data, size, mem_pool[gpu_index]));
        
        // Create CUFFT plan
        CHECK_CUDA(cufftCreate(&ctx.plan));
        CHECK_CUDA(cufftSetStream(ctx.plan, ctx.stream));
    }
    
    void cleanup() {
        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(contexts[i].device_id);
            cufftDestroy(contexts[i].plan);
            cudaFree(contexts[i].d_data);
            cudaStreamDestroy(contexts[i].stream);
            cudaMemPoolDestroy(mem_pool[i]);
        }
        delete[] contexts;
    }

public:
    void execute(float2* h_data, float2* h_result) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        #pragma omp parallel num_threads(num_gpus)
        {
            int gpu_id = omp_get_thread_num();
            GPUContext& ctx = contexts[gpu_id];
            
            // Set device for this thread
            CHECK_CUDA(cudaSetDevice(ctx.device_id));
            
            // Calculate offset and size for this GPU's chunk
            size_t offset = 0;
            for (int i = 0; i < gpu_id; i++) {
                offset += contexts[i].chunk_size;
            }
            
            // Profile memory transfer
            nvtxRangePush("Memory Transfer to GPU");
            CHECK_CUDA(cudaMemcpyAsync(ctx.d_data, 
                                     h_data + offset * width,
                                     ctx.chunk_size * width * sizeof(float2),
                                     cudaMemcpyHostToDevice, 
                                     ctx.stream));
            nvtxRangePop();
            
            // Execute row-wise FFT
            nvtxRangePush("Row FFT");
            executeRowFFT(ctx);
            nvtxRangePop();
            
            // Synchronize for transpose
            #pragma omp barrier
            
            // Global transpose and column FFT
            nvtxRangePush("Column FFT");
            executeColumnFFT(ctx);
            nvtxRangePop();
            
            // Transfer results back
            nvtxRangePush("Memory Transfer to Host");
            CHECK_CUDA(cudaMemcpyAsync(h_result + offset * width,
                                     ctx.d_data,
                                     ctx.chunk_size * width * sizeof(float2),
                                     cudaMemcpyDeviceToHost,
                                     ctx.stream));
            nvtxRangePop();
        }
        
        // Record timing
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        updateMetrics(milliseconds);
    }
    
private:
    void executeRowFFT(GPUContext& ctx) {
        dim3 block(MAX_BLOCK_SIZE);
        dim3 grid((ctx.chunk_size + block.x - 1) / block.x);
        
        for (int stage = 1; stage <= log2(width); stage++) {
            fft_butterfly_rows_optimized_v2<<<grid, block, 0, ctx.stream>>>(
                ctx.d_data, width, ctx.chunk_size, stage, 1 << (stage - 1));
        }
    }
    
    void executeColumnFFT(GPUContext& ctx) {
        dim3 block(MAX_BLOCK_SIZE);
        dim3 grid((width + block.x - 1) / block.x);
        
        for (int stage = 1; stage <= log2(height); stage++) {
            fft_butterfly_columns_optimized_v2<<<grid, block, 0, ctx.stream>>>(
                ctx.d_data, width, ctx.chunk_size, stage, 1 << (stage - 1));
        }
    }
    
    void updateMetrics(float milliseconds) {
        metrics.total_time = milliseconds / 1000.0f;
        metrics.throughput = (width * height * sizeof(float2)) / 
                           (metrics.total_time * 1e9); // GB/s
    }
    
public:
    PerfMetrics getPerformanceMetrics() const {
        return metrics;
    }
};

// Main execution function
void compute_2d_multi_gpu(float* buffer, int rows, int cols, int sample_rate, const char* filename) {
    // Convert input to complex format
    float2* h_data = new float2[rows * cols];
    for (int i = 0; i < rows * cols; i++) {
        h_data[i].x = buffer[i];
        h_data[i].y = 0.0f;
    }
    
    float2* h_result = new float2[rows * cols];
    
    // Create and execute multi-GPU FFT
    FFT2DMultiGPU fft(cols, rows);
    fft.execute(h_data, h_result);
    
    // Get performance metrics
    PerfMetrics metrics = fft.getPerformanceMetrics();
    printf("Performance Metrics:\n");
    printf("Total Time: %.3f seconds\n", metrics.total_time);
    printf("Throughput: %.2f GB/s\n", metrics.throughput);
    
    // Save results
    saveResults(h_result, rows, cols, filename);
    
    delete[] h_data;
    delete[] h_result;
}