#include "common.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <dirent.h>  // For directory handling
#include <cstring>

using namespace std;

#define M_PI 3.14159265358979323846

// Function for 1D FFT on rows
void fft_1d_row(float* data, int width, int height, int step) {
    int half_step = step / 2;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < half_step; ++col) {
            int idx1 = row * width * 2 + col * 2;
            int idx2 = idx1 + step * 2;

            float angle = -2.0f * M_PI * col / step;
            float t_real = cosf(angle);
            float t_imag = sinf(angle);

            float u_real = data[idx1];
            float u_imag = data[idx1 + 1];
            float v_real = data[idx2];
            float v_imag = data[idx2 + 1];

            float tr = t_real * v_real - t_imag * v_imag;
            float ti = t_real * v_imag + t_imag * v_real;

            data[idx1] = u_real + tr;
            data[idx1 + 1] = u_imag + ti;
            data[idx2] = u_real - tr;
            data[idx2 + 1] = u_imag - ti;
        }
    }
}

// Function for 1D FFT on columns
void fft_1d_column(float* data, int width, int height, int step) {
    int half_step = step / 2;
    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < half_step; ++row) {
            int idx1 = col * 2 + row * width * 2;
            int idx2 = idx1 + step * width * 2;

            float angle = -2.0f * M_PI * row / step;
            float t_real = cosf(angle);
            float t_imag = sinf(angle);

            float u_real = data[idx1];
            float u_imag = data[idx1 + 1];
            float v_real = data[idx2];
            float v_imag = data[idx2 + 1];

            float tr = t_real * v_real - t_imag * v_imag;
            float ti = t_real * v_imag + t_imag * v_real;

            data[idx1] = u_real + tr;
            data[idx1 + 1] = u_imag + ti;
            data[idx2] = u_real - tr;
            data[idx2 + 1] = u_imag - ti;
        }
    }
}



void compute_2d_fft_cpu(float* data, int width, int height) {
    // Perform FFT on rows
    for (int step = 2; step <= width; step <<= 1) {
        fft_1d_row(data, width, height, step);
    }
    // Perform FFT on columns
    for (int step = 2; step <= height; step <<= 1) {
        fft_1d_column(data, width, height, step);
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
    std::vector<float> data;
    if (!read_matrix_data(argv[1], data, width, height)) {
        return 1;
    }

    // Print input sample
    std::cout << "Input Matrix (partial):" << std::endl;
    for (int i = 0; i < std::min(5, height); ++i) {
        for (int j = 0; j < std::min(5, width); ++j) {
            std::cout << data[i * width * 2 + j * 2] << " ";
        }
        std::cout << std::endl;
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