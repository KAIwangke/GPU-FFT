#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <filesystem>

void generate_matrix(int size) {
    std::string filename = std::to_string(size) + "x" + std::to_string(size) + "_matrix.dat";
    std::ofstream outfile(filename);
    
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    
    // Generate random matrix data and write to file
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float real_part = static_cast<float>(rand() % 100);
            outfile << real_part << " ";
        }
        outfile << "\n";
    }
    outfile.close();
    std::cout << size << "x" << size << " random matrix written to " << filename << "\n";
}

int main() {
    // Seed random number generator
    srand(time(NULL));
    
    // Create input directory if it doesn't exist
    std::filesystem::create_directory("input");
    
    // Generate matrices for different sizes
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    for (int size : sizes) {
        generate_matrix(size);
    }
    
    return 0;
}