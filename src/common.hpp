#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <iostream>
#include <cstring>
using namespace std;


#define M_PI 3.14159265358979323846

// Error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Function to determine matrix dimensions from file
void get_matrix_dimensions(const char* filename, int& width, int& height) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    // Read first line to determine width
    if (getline(file, line)) {
        std::istringstream iss(line);
        float value;
        width = 0;
        while (iss >> value) {
            ++width;
        }
    }

    // Count newlines to determine height
    height = 1;
    while (getline(file, line)) {
        ++height;
    }

    file.close();
}

// Function to read matrix data from file
bool read_matrix_data(const char* filename, std::vector<float>& data, int width, int height) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return false;
    }

    data.resize(width * height * 2); // *2 for complex numbers
    for (int i = 0; i < width * height; i++) {
        file >> data[i * 2];     // Real part
        data[i * 2 + 1] = 0.0f;  // Imaginary part set to 0
    }

    file.close();
    return true;
}
