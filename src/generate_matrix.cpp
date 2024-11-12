#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <filesystem>

void generate_matrix(int size, const std::string& base_dir) {
    std::string input_dir = base_dir + "/input";
    std::filesystem::create_directories(input_dir);  // Create nested directories if needed
    
    std::string filename = input_dir + "/" + std::to_string(size) + "x" + std::to_string(size) + "_matrix.dat";
    
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
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size> [base_directory]\n";
        return 1;
    }
    
    srand(time(NULL));
    
    int size = std::atoi(argv[1]);
    std::string base_dir = (argc > 2) ? argv[2] : ".";
    
    generate_matrix(size, base_dir);
    return 0;
}