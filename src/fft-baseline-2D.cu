#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <dirent.h>
#include <cstring>

using namespace std;

// CUDA Kernel for bit-reversal reordering on each row
__global__ void bit_reverse_reorder_rows(float* d_r, float* d_x, int width, int height, int s) {
    int row = blockIdx.x;
    int i = threadIdx.x;
    if (row < height && i < width) {
        int j = i, k = 0;
        for (int l = 1; l <= s; l++) {
            k = k * 2 + (j & 1);
            j >>= 1;
        }
        d_r[row * width * 2 + k * 2] = d_x[row * width + i];
        d_r[row * width * 2 + k * 2 + 1] = 0;
    }
}

// CUDA Kernel for butterfly computation on rows
__global__ void fft_butterfly_rows(float* d_r, int width, int height, int m) {
    int row = blockIdx.x;
    int k = threadIdx.x;
    if (row < height && k < m / 2) {
        int ridx = row * width * 2 + k * 2;
        int ridx2 = ridx + m;
        
        float angle = (2.0f * M_PI * k) / m;
        float t_real = cosf(angle);
        float t_imag = -sinf(angle);

        float u_real = d_r[ridx];
        float u_imag = d_r[ridx + 1];

        float tr_real = t_real * d_r[ridx2] - t_imag * d_r[ridx2 + 1];
        float tr_imag = t_real * d_r[ridx2 + 1] + t_imag * d_r[ridx2];

        d_r[ridx] = u_real + tr_real;
        d_r[ridx + 1] = u_imag + tr_imag;

        d_r[ridx2] = u_real - tr_real;
        d_r[ridx2 + 1] = u_imag - tr_imag;
    }
}

// CUDA Kernel for butterfly computation on columns
__global__ void fft_butterfly_columns(float* d_r, int width, int height, int m) {
    int col = blockIdx.x;
    int k = threadIdx.x;
    if (col < width && k < m / 2) {
        int ridx = col * 2 + k * 2 * width;
        int ridx2 = ridx + m * width * 2;

        float angle = (2.0f * M_PI * k) / m;
        float t_real = cosf(angle);
        float t_imag = -sinf(angle);

        float u_real = d_r[ridx];
        float u_imag = d_r[ridx + 1];

        float tr_real = t_real * d_r[ridx2] - t_imag * d_r[ridx2 + 1];
        float tr_imag = t_real * d_r[ridx2 + 1] + t_imag * d_r[ridx2];

        d_r[ridx] = u_real + tr_real;
        d_r[ridx + 1] = u_imag + tr_imag;

        d_r[ridx2] = u_real - tr_real;
        d_r[ridx2 + 1] = u_imag - tr_imag;
    }
}

// 2D FFT function
void fft_2d(float* d_data, int width, int height) {
    int s = log2(width);

    // Launch row-wise FFT
    for (int i = 1; i <= s; i++) {
        int m = 1 << i;
        fft_butterfly_rows<<<height, width / 2>>>(d_data, width, height, m);
        cudaDeviceSynchronize();
    }

    // Launch column-wise FFT
    s = log2(height);
    for (int i = 1; i <= s; i++) {
        int m = 1 << i;
        fft_butterfly_columns<<<width, height / 2>>>(d_data, width, height, m);
        cudaDeviceSynchronize();
    }
}

// Reads 2D matrix data from a file
void read_file(const char* filename, vector<vector<float>>& matrix, int& rows, int& cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Can't open file " << filename << " for reading." << endl;
        return;
    }

    matrix.clear();
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        float value;
        vector<float> row;
        while (ss >> value) {
            row.push_back(value);
        }
        if (cols == 0) {
            cols = row.size(); // Set column size based on first row
        }
        matrix.push_back(row);
    }
    rows = matrix.size();
    file.close();
}

// Flatten matrix for CUDA processing
void flatten_matrix(const vector<vector<float>>& matrix, vector<float>& flat_matrix) {
    for (const auto& row : matrix) {
        flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }
}

// 2D FFT Compute function
#include <chrono>

void compute_2d(float* buffer, int rows, int cols, int sample_rate, const char* filename) {
    float* result;

    // Allocate memory for the result on the host
    result = new float[rows * cols * 2]; // 2x for complex data (real and imaginary parts)

    // Allocate memory on the device for the input buffer and result
    float* d_buffer;
    cudaMalloc((void**)&d_buffer, rows * cols * 2 * sizeof(float)); // Complex data, so 2 * sizeof(float)

    // Start the stopwatch (includes memory copy to device)
    auto start = chrono::high_resolution_clock::now();

    // Copy data to the device
    cudaMemcpy(d_buffer, buffer, rows * cols * 2 * sizeof(float), cudaMemcpyHostToDevice);

    // Run 2D FFT on GPU
    fft_2d(d_buffer, cols, rows);

    // Copy result back to the host
    cudaMemcpy(result, d_buffer, rows * cols * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Stop the stopwatch (after memory copy from device)
    auto finish = chrono::high_resolution_clock::now();
    auto microseconds = chrono::duration_cast<std::chrono::microseconds>(finish - start);
    cout << "Elapsed time (including memory transfer): " << microseconds.count() / 1e6 << "seconds" << endl;

    // Free device memory
    cudaFree(d_buffer);

    // Save the computed data with complex values
    char outfilename[512];
    strcpy(outfilename, filename);
    strcat(outfilename, ".out");
    ofstream outfile;
    outfile.open(outfilename);
    outfile.precision(6);

    // Print results in a format similar to the CPU version
    outfile << "2D FFT Output (partial):" << endl;
    for (int i = 0; i < min(5, rows); i++) {
        for (int j = 0; j < min(5, cols); j++) {
            float real_part = result[(i * cols + j) * 2];
            float imag_part = result[(i * cols + j) * 2 + 1];
            outfile << "X[" << i << "][" << j << "] = " << real_part << " + " << imag_part << "i\t";
        }
        outfile << endl;
    }

    outfile.close();
    delete[] result;
}

int main(int argc, char** argv) {
    srand(time(NULL));

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " [input_folder]";
        return 2;
    }

    DIR* dirp = opendir(argv[1]);
    struct dirent* epdf;
    while ((epdf = readdir(dirp)) != NULL) {
        size_t len = strlen(epdf->d_name);
        if (strcmp(epdf->d_name, ".") != 0 && strcmp(epdf->d_name, "..") != 0
            && strcmp(&epdf->d_name[len - 3], "dat") == 0) {
            stringstream fname(epdf->d_name);
            string samples, sr;
            
            getline(fname, samples, '@');
            getline(fname, sr, '.');
            
            char fold[512];
            strcpy(fold, argv[1]);
            
            vector<vector<float>> matrix;
            int rows = 0, cols = 0;
            read_file(strcat(strcat(fold, "/"), epdf->d_name), matrix, rows, cols);
            
            vector<float> flat_matrix;
            flatten_matrix(matrix, flat_matrix);
            compute_2d(&flat_matrix[0], rows, cols, atoi(sr.c_str()), epdf->d_name);
        }
    }
    
    closedir(dirp);
    return 0;
}
