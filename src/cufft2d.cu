#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

// Read matrix data from a file
bool read_matrix_from_file(const string& filename, vector<float>& matrix, int rows, int cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << filename << endl;
        return false;
    }

    matrix.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        file >> matrix[i];
        if (file.fail()) {
            cerr << "Error: Failed to read data from file." << endl;
            file.close();
            return false;
        }
    }

    file.close();
    return true;
}

// Perform real-to-complex 2D FFT (R2C) using cuFFT and measure total execution time including data transfer
void cufft_2d_r2c(const vector<float>& h_data, int rows, int cols) {
    int complex_cols = cols / 2 + 1;

    // Allocate device memory for input and output data
    cufftReal* d_data_in;
    cufftComplex* d_data_out;
    if (cudaMalloc((void**)&d_data_in, sizeof(cufftReal) * rows * cols) != cudaSuccess) {
        cerr << "Error allocating d_data_in" << endl;
        return;
    }
    if (cudaMalloc((void**)&d_data_out, sizeof(cufftComplex) * rows * complex_cols) != cudaSuccess) {
        cerr << "Error allocating d_data_out" << endl;
        cudaFree(d_data_in);
        return;
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Copy data from host to device
    if (cudaMemcpy(d_data_in, h_data.data(), sizeof(cufftReal) * rows * cols, cudaMemcpyHostToDevice) != cudaSuccess) {
        cerr << "Error copying data to device" << endl;
        cudaFree(d_data_in);
        cudaFree(d_data_out);
        return;
    }

    // Create cuFFT 2D plan
    cufftHandle plan;
    if (cufftPlan2d(&plan, rows, cols, CUFFT_R2C) != CUFFT_SUCCESS) {
        cerr << "CUFFT plan creation failed" << endl;
        cudaFree(d_data_in);
        cudaFree(d_data_out);
        return;
    }

    // Execute R2C FFT
    if (cufftExecR2C(plan, d_data_in, d_data_out) != CUFFT_SUCCESS) {
        cerr << "CUFFT execution failed" << endl;
        cufftDestroy(plan);
        cudaFree(d_data_in);
        cudaFree(d_data_out);
        return;
    }

    // Copy results from device back to host
    vector<cufftComplex> h_result(rows * complex_cols);
    if (cudaMemcpy(h_result.data(), d_data_out, sizeof(cufftComplex) * rows * complex_cols, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cerr << "Error copying data from device" << endl;
        cufftDestroy(plan);
        cudaFree(d_data_in);
        cudaFree(d_data_out);
        return;
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate total execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Total Execution Time (including cudaMemcpy): " << milliseconds << " ms" << endl;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print partial result
    cout << "Partial FFT Output (complex values):" << endl;
    for (int i = 0; i < min(5, rows); ++i) {
        for (int j = 0; j < min(5, complex_cols); ++j) {
            cout << "(" << h_result[i * complex_cols + j].x << ", " << h_result[i * complex_cols + j].y << ") ";
        }
        cout << endl;
    }

    // Destroy cuFFT plan and free memory
    cufftDestroy(plan);
    cudaFree(d_data_in);
    cudaFree(d_data_out);
}

int main() {
    int rows = 1000;    // Number of matrix rows
    int cols = 1000;    // Number of matrix columns

    vector<float> h_data;  // Host storage for the real matrix

    // Read matrix data from file
    if (!read_matrix_from_file("1000doubleTransposed.dat", h_data, rows, cols)) {
        return 1;  // Exit program if file reading fails
    }

    // Print partial content of the matrix to confirm data correctness
    cout << "Input Matrix (partial):" << endl;
    for (int i = 0; i < min(5, rows); ++i) {
        for (int j = 0; j < min(5, cols); ++j) {
            cout << h_data[i * cols + j] << " ";
        }
        cout << endl;
    }

    // Normalize data
    for (auto& val : h_data) {
        val /= 100.0f;
    }

    // Run cuFFT 2D R2C FFT function
    cout << "Running CUFFT R2C 2D FFT on real input..." << endl;
    cufft_2d_r2c(h_data, rows, cols);

    return 0;
}


/* 
scp /Users/lidanwen/Desktop/gpu/project/1000doubleTransposed.dat /Users/lidanwen/Desktop/gpu/project/1000doubleTransposed.dat dl5179@access.cims.nyu.edu:~/gpu/project

scp /Users/lidanwen/Desktop/gpu/project/cufft2d.cu dl5179@access.cims.nyu.edu:~/gpu/project
6AFwk?J*ZGSx#m
ssh dl5179@access.cims.nyu.edu
6AFwk?J*ZGSx#m
ssh cuda2
6AFwk?J*ZGSx#m
module load cuda-12.4
cd ~/gpu/project
nvcc -o cufft2d cufft2d.cu -lcufft
./cufft2d
*/