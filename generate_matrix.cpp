#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

#define NX 256
#define NY 256

int main() {
    std::ofstream outfile("64x64_matrix.dat");
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file for writing.\n";
        return 1;
    }

    // Seed random number generator
    srand(time(NULL));

    // Generate random matrix data and write to file
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            float real_part = static_cast<float>(rand() % 100);
            outfile << real_part << " ";
        }
        outfile << "\n";  // New line after each row
    }

    outfile.close();
    std::cout << "64x64 random matrix written to 64x64_matrix.dat\n";
    return 0;
}
