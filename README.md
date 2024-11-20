
# GPU_FFT Project

The GPU_FFT project is designed to perform fast Fourier transforms (FFT) on large matrices using CUDA. It includes implementations that leverage different optimization strategies across multiple stages of the FFT computation.

## Requirements

- CMake 3.18 or higher
- CUDA Toolkit (Compatible with your GPU architecture)
- A compatible C++ compiler
- NVIDIA GPU with Compute Capability 7.5 or higher (for specified architectures)

## Building the Project

To build the project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a build directory and navigate into it:

   ```bash
   mkdir build && cd build
   ```

4. Run CMake configuration and then build the project:

   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make
   ```

This will compile all the necessary executables for both generating matrices and performing FFT operations.

## Running the Project

To run FFT operations:

1. Generate input matrices:

   ```bash
   make create_all_inputs
   ```

2. Execute FFT on a specific matrix size:

   ```bash
   make run_all_1024  # Example for 1024x1024 matrix size
   ```

You can run the project for different stages and sizes by modifying the size in the command.

Below are the `make` commands for running the FFT implementations for all predefined matrix sizes in your project. These commands will execute the FFT on each size for each implementation stage as well as the CuFFT library, assuming the input matrix data files have already been generated.

### Run Commands for Each FFT Implementation and Matrix Size

1. **CPU Implementation:**
   ```bash
   make run_cpu_32
   make run_cpu_64
   make run_cpu_128
   make run_cpu_256
   make run_cpu_512
   make run_cpu_1024
   make run_cpu_2048
   make run_cpu_4096
   make run_cpu_8192
   make run_cpu_16384
   ```

2. **Stage 0 GPU Implementation:**
   ```bash
   make run_stage0_N
   ```

3. **Stage 1 Shared memory optimized kernel**
   ```bash
   make run_stage1_N
   ```

4. **Stage 2 Memory coalescing and bank conflict optimization**
   ```bash
   make run_stage2_N
   ```

5. **Stage 3 Warp-level optimization**
   ```bash
   make run_stage3_N
   ```

6. **CuFFT Library:**
   ```bash
   make run_cufft_N
   ```

### Run All FFT Implementations for a Specific Matrix Size

If you want to run all FFT implementations for a specific matrix size, you can use commands like the following:

```bash
make run_all_N
```

These commands will trigger all available implementations (CPU, all GPU stages, and CuFFT) for each respective matrix size. Make sure all input matrices are generated before running these commands to avoid errors.

## Performance Analysis

This project is equipped with NVIDIA's `ncu` tool commands integrated into custom targets for memory transactions, throughput, and cache efficiency analysis. To execute these analyses for a specific implementation (e.g., `fft_stage0`), run:

```bash
make stage0_analysis
```

This will output CSV files containing performance metrics to the `eval_results` directory.

## Benchmarks

To benchmark all small or large matrix sizes, use:

```bash
make benchmark_small
make benchmark_large
```

These commands will run FFTs on all predefined small and large matrix sizes respectively.

## Comprehensive Report

To generate a comprehensive performance report combining all CSV files:

```bash
make generate_report
```

This will concatenate all analysis results into a single CSV file in the `eval_results` directory.

## Additional Notes

- Ensure CUDA and OpenMP packages are correctly installed and configured on your system.
- The project uses CUDA architecture 75 by default, which corresponds to NVIDIA Turing architecture GPUs. Adjust the `CMAKE_CUDA_ARCHITECTURES` in the CMakeLists file to match your GPU capabilities.