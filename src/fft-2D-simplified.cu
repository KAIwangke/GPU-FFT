#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <dirent.h>  // For directory handling
#include <cstring>

using namespace std;

#define M_PI 3.14159265358979323846

// CUDA Kernel for 1D FFT on rows
__global__ void fft_1d_row(float* d_r, int width, int height, int step) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    int half_step = step / 2;
    if (col < half_step && row < height) {
        int idx1 = row * width * 2 + col * 2;
        int idx2 = idx1 + step * 2;

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
}

// CUDA Kernel for 1D FFT on columns
__global__ void fft_1d_column(float* d_r, int width, int height, int step) {
    int col = blockIdx.x;
    int row = threadIdx.x;

    int half_step = step / 2;
    if (row < half_step && col < width) {
        int idx1 = col * 2 + row * width * 2;
        int idx2 = idx1 + step * width * 2;

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
}

// Function to determine matrix dimensions from file
void get_matrix_dimensions(const char* filename, int &width, int &height) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read first line to determine width
    std::string line;
    if (getline(file, line)) {
        std::istringstream iss(line);
        float value;
        width = 0;
        while (iss >> value) {
            ++width;
        }
    }

    // Count newlines to determine height
    height = 1; // We already read one line
    while (getline(file, line)) {
        ++height;
    }

    file.close();
}

// 2D FFT function (unchanged)
void fft_2d(float* d_data, int width, int height) {
    for (int step = 2; step <= width; step <<= 1) {
        fft_1d_row<<<height, width / 2>>>(d_data, width, height, step);
        cudaDeviceSynchronize();
    }
    for (int step = 2; step <= height; step <<= 1) {
        fft_1d_column<<<width, height / 2>>>(d_data, width, height, step);
        cudaDeviceSynchronize();
    }
}

// Main function
int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_directory>" << endl;
        return 1;
    }

    const char* directory_path = argv[1];
    DIR* dir = opendir(directory_path);
    if (!dir) {
        cerr << "Error: Cannot open directory " << directory_path << endl;
        return 1;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        // Process only .dat files
        if (strstr(entry->d_name, ".dat") != NULL) {
            string input_filepath = string(directory_path) + "/" + entry->d_name;

            // Determine matrix dimensions
            int width, height;
            get_matrix_dimensions(input_filepath.c_str(), width, height);
            cout << "Processing " << entry->d_name << " with dimensions: " << width << "x" << height << endl;

            int size = width * height * 2;
            float* h_data = new float[size];

            // Read matrix data from file
            ifstream infile(input_filepath);
            if (!infile.is_open()) {
                cerr << "Error: Cannot open file " << input_filepath << endl;
                delete[] h_data;
                continue;
            }
            for (int i = 0; i < width * height; i++) {
                infile >> h_data[i * 2];    // Real part
                h_data[i * 2 + 1] = 0.0f;   // Imaginary part set to 0
            }
            infile.close();

            // Allocate device memory
            float* d_data;
            cudaMalloc(&d_data, size * sizeof(float));

            // Start timing
            auto start = chrono::high_resolution_clock::now();

            // Copy data to device
            cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

            // Perform 2D FFT on GPU
            fft_2d(d_data, width, height);

            // Copy result back to host
            cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

            // Stop timing
            auto finish = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(finish - start).count();
            cout << "Elapsed time for " << entry->d_name << " (including memcpy): " << duration / 1e6 << " seconds" << endl;

            // Create output filename
            string output_filename = string(entry->d_name) + ".out";
            ofstream outfile(output_filename);
            outfile.precision(6);
            outfile << "2D FFT Output (partial):" << endl;
            for (int i = 0; i < min(5, height); i++) {
                for (int j = 0; j < min(5, width); j++) {
                    float real_part = h_data[(i * width + j) * 2];
                    float imag_part = h_data[(i * width + j) * 2 + 1];
                    outfile << "X[" << i << "][" << j << "] = " << real_part << " + " << imag_part << "i\t";
                }
                outfile << endl;
            }
            outfile.close();

            // Free memory
            delete[] h_data;
            cudaFree(d_data);
        }
    }
    closedir(dir);
    return 0;
}