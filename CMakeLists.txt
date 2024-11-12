cmake_minimum_required(VERSION 3.18)
project(GPU_FFT CUDA CXX)

# Set CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Find CUDA
find_package(CUDA REQUIRED)

# Set compiler flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Add executables
add_executable(fft_cpu src/fft-cpu-2D.cc)
add_executable(fft_stage0 src/fft-stage0-2D.cu)
add_executable(fft_stage1 src/fft-stage1-2D.cu)
add_executable(fft_stage2 src/fft-stage2-2D.cu)
add_executable(fft_stage3 src/fft-stage3-2D.cu)
add_executable(fft_cufft src/cufft2d.cu)

# Link CUDA libraries
target_link_libraries(fft_cufft PRIVATE cufft)

# Add run targets
foreach(size 64 256 1000)
    foreach(target cpu stage0 stage1 stage2 stage3 cufft)
        add_custom_target(run_${target}_${size}
            COMMAND fft_${target} ${CMAKE_SOURCE_DIR}/input/${size}.dat
            DEPENDS fft_${target}
        )
    endforeach()
endforeach()

# Add combined run targets
add_custom_target(run_all
    DEPENDS
        run_cpu_64 run_cpu_256 run_cpu_1000
        run_stage0_64 run_stage0_256 run_stage0_1000
        run_stage1_64 run_stage1_256 run_stage1_1000
        run_stage2_64 run_stage2_256 run_stage2_1000
        run_stage3_64 run_stage3_256 run_stage3_1000
        run_cufft_64 run_cufft_256 run_cufft_1000
)