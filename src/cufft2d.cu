#include "common.hpp"
#include <cufft.h>

void compute_2d_fft_cufft(const std::vector<float>& h_data, int width, int height, 
                         std::vector<cufftComplex>& h_result) {
    int complex_cols = width / 2 + 1;
    
    // Allocate device memory
    cufftReal* d_data_in;
    cufftComplex* d_data_out;
    CHECK_CUDA(cudaMalloc((void**)&d_data_in, sizeof(cufftReal) * width * height));
    CHECK_CUDA(cudaMalloc((void**)&d_data_out, sizeof(cufftComplex) * height * complex_cols));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Start timing
    CHECK_CUDA(cudaEventRecord(start));

    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(d_data_in, h_data.data(), 
                         sizeof(cufftReal) * width * height, 
                         cudaMemcpyHostToDevice));

    // Create cuFFT plan
    cufftHandle plan;
    if (cufftPlan2d(&plan, height, width, CUFFT_R2C) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT plan creation failed" << std::endl;
        cudaFree(d_data_in);
        cudaFree(d_data_out);
        return;
    }

    // Execute FFT
    if (cufftExecR2C(plan, d_data_in, d_data_out) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT execution failed" << std::endl;
        cufftDestroy(plan);
        cudaFree(d_data_in);
        cudaFree(d_data_out);
        return;
    }

    // Copy results back to host
    h_result.resize(height * complex_cols);
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_data_out,
                         sizeof(cufftComplex) * height * complex_cols,
                         cudaMemcpyDeviceToHost));

    // Stop timing
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Total execution time: " << milliseconds << " ms" << std::endl;

    // Cleanup
    cufftDestroy(plan);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data_in);
    cudaFree(d_data_out);
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
    std::vector<float> h_data_in;
    if (!read_matrix_data(argv[1], h_data_in, width, height)) {
        return 1;
    }

    // Convert input to single precision real
    std::vector<float> h_data_real(width * height);
    for (int i = 0; i < width * height; ++i) {
        h_data_real[i] = h_data_in[i * 2] / 100.0f;  // Normalize input
    }

    // Perform FFT
    std::vector<cufftComplex> h_result;
    compute_2d_fft_cufft(h_data_real, width, height, h_result);



    return 0;
}