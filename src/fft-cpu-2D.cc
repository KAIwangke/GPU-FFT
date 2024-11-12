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

// 2D FFT function
void fft_2d(float* data, int width, int height) {
    // Perform FFT on rows
    for (int step = 2; step <= width; step <<= 1) {
        fft_1d_row(data, width, height, step);
    }
    // Perform FFT on columns
    for (int step = 2; step <= height; step <<= 1) {
        fft_1d_column(data, width, height, step);
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

            // Start timing
            auto start = chrono::high_resolution_clock::now();

            // Perform 2D FFT on CPU
            fft_2d(h_data, width, height);

            // Stop timing
            auto finish = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(finish - start).count();
            cout << "Elapsed time for " << entry->d_name << ": " << duration / 1e6 << " seconds" << endl;

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
        }
    }
    closedir(dir);
    return 0;
}
