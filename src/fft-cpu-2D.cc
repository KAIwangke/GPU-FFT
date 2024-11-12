#include "common.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define M_PI 3.14159265358979323846

// Simple 1D FFT implementation
void fft_1d(float* real, float* imag, int n) {
    // Base case
    if (n <= 1) return;

    std::vector<float> even_real(n/2), even_imag(n/2);
    std::vector<float> odd_real(n/2), odd_imag(n/2);
    
    for (int i = 0; i < n/2; i++) {
        even_real[i] = real[2*i];
        even_imag[i] = imag[2*i];
        odd_real[i] = real[2*i + 1];
        odd_imag[i] = imag[2*i + 1];
    }

    // Recursive FFT on even and odd parts
    fft_1d(even_real.data(), even_imag.data(), n/2);
    fft_1d(odd_real.data(), odd_imag.data(), n/2);

    // Combine results
    for (int k = 0; k < n/2; k++) {
        float angle = -2.0f * M_PI * k / n;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        
        float t_real = cos_val * odd_real[k] - sin_val * odd_imag[k];
        float t_imag = cos_val * odd_imag[k] + sin_val * odd_real[k];
        
        real[k] = even_real[k] + t_real;
        imag[k] = even_imag[k] + t_imag;
        
        real[k + n/2] = even_real[k] - t_real;
        imag[k + n/2] = even_imag[k] - t_imag;
    }
}

void compute_2d_fft_cpu(float* data, int width, int height) {
    std::vector<float> real_row(width), imag_row(width);
    std::vector<float> real_col(height), imag_col(height);
    
    // Process rows
    for (int i = 0; i < height; i++) {
        // Extract row
        for (int j = 0; j < width; j++) {
            real_row[j] = data[(i * width + j) * 2];
            imag_row[j] = data[(i * width + j) * 2 + 1];
        }
        
        // FFT on row
        fft_1d(real_row.data(), imag_row.data(), width);
        
        // Write back row
        for (int j = 0; j < width; j++) {
            data[(i * width + j) * 2] = real_row[j];
            data[(i * width + j) * 2 + 1] = imag_row[j];
        }
    }
    
    // Process columns
    for (int j = 0; j < width; j++) {
        // Extract column
        for (int i = 0; i < height; i++) {
            real_col[i] = data[(i * width + j) * 2];
            imag_col[i] = data[(i * width + j) * 2 + 1];
        }
        
        // FFT on column
        fft_1d(real_col.data(), imag_col.data(), height);
        
        // Write back column
        for (int i = 0; i < height; i++) {
            data[(i * width + j) * 2] = real_col[i];
            data[(i * width + j) * 2 + 1] = imag_col[i];
        }
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
    
    // Verify dimensions are power of 2
    if ((width & (width - 1)) != 0 || (height & (height - 1)) != 0) {
        std::cerr << "Error: Matrix dimensions must be power of 2" << std::endl;
        return 1;
    }
    
    std::cout << "Processing matrix of size " << width << "x" << height << std::endl;

    // Read input data
    std::vector<float> data;
    if (!read_matrix_data(argv[1], data, width, height)) {
        return 1;
    }

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Perform FFT
    compute_2d_fft_cpu(data.data(), width, height);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Total execution time: " << duration.count() << " ms" << std::endl;
    return 0;
}