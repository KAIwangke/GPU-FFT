#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <filesystem>

void generate_matrix(int size) {
    std::filesystem::create_directory("input");
    std::string filename = "input/" + std::to_string(size) + "x" + std::to_string(size) + "_matrix.dat";
    
    // Check if file already exists
    if (std::filesystem::exists(filename)) {
        std::cout << "Matrix " << filename << " already exists, skipping.\n";
        return;
    }

    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    
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

int main(int argc, char** argv) {
    srand(time(NULL));
    
    #ifdef SINGLE_SIZE_MODE
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <size>\n";
        return 1;
    }
    generate_matrix(std::atoi(argv[1]));
    #else
    // Original multiple size generation code
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    for (int size : sizes) {
        generate_matrix(size);
    }
    #endif

    return 0;
}