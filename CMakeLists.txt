cmake_minimum_required(VERSION 3.18)
project(GPU_FFT CUDA CXX)

# Set CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Find CUDA and OpenMP
find_package(CUDA REQUIRED)
find_package(OpenMP)

# Set compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Define directories
set(INPUT_DIR ${CMAKE_SOURCE_DIR}/input)
set(EVAL_DIR ${CMAKE_BINARY_DIR}/eval_results)

# Add matrix generator executable
add_executable(generate_matrix src/generate_matrix.cpp)

# Add FFT executables
add_executable(fft_cpu src/fft-cpu-2D.cc)
add_executable(fft_stage0 src/fft-stage0-2D.cu)
add_executable(fft_stage1 src/fft-stage1-2D.cu)
add_executable(fft_stage2 src/fft-stage2-2D.cu)
add_executable(fft_stage3 src/fft-stage3-2D.cu)
add_executable(fft_cufft src/cufft2d.cu)

# Link libraries
target_link_libraries(fft_cufft PRIVATE cufft)
target_link_libraries(generate_matrix PRIVATE stdc++fs)

# Set CUDA properties
foreach(target fft_stage0 fft_stage1 fft_stage2 fft_stage3 fft_cufft)
    set_target_properties(${target} PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()

# Create input directory and generate matrices
add_custom_target(create_inputs
    COMMAND ${CMAKE_BINARY_DIR}/generate_matrix
    DEPENDS generate_matrix
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Define matrix sizes
set(MATRIX_SIZES 32 64 128 256 512 1024 2048 4096 8192 16384)

# Add run targets for each implementation and size
foreach(size ${MATRIX_SIZES})
    foreach(impl cpu stage0 stage1 stage2 stage3 cufft)
        add_custom_target(run_${impl}_${size}
            COMMAND fft_${impl} ${INPUT_DIR}/${size}x${size}_matrix.dat
            DEPENDS fft_${impl} create_inputs
        )
    endforeach()
    
    # Add combined target for each size
    add_custom_target(run_all_${size}
        DEPENDS 
            run_cpu_${size}
            run_stage0_${size}
            run_stage1_${size}
            run_stage2_${size}
            run_stage3_${size}
            run_cufft_${size}
    )
endforeach()

# Add performance analysis targets
foreach(impl stage0 stage1 stage2 stage3 cufft)
    # Memory transaction analysis
    add_custom_target(${impl}_memtrans
        COMMAND ncu --metrics "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
        l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
        l1tex__t_sector_hit_rate.pct,\
        lts__t_sector_hit_rate.pct"
        $<TARGET_FILE:fft_${impl}> ${INPUT_DIR}/1024x1024_matrix.dat
        DEPENDS fft_${impl} create_inputs
    )
    
    # Memory throughput analysis
    add_custom_target(${impl}_throughput
        COMMAND ncu --metrics "dram__bytes_read.sum,\
        dram__bytes_write.sum,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed"
        $<TARGET_FILE:fft_${impl}> ${INPUT_DIR}/1024x1024_matrix.dat
        DEPENDS fft_${impl} create_inputs
    )
    
    # Cache efficiency analysis
    add_custom_target(${impl}_cache
        COMMAND ncu --metrics "l1tex__t_hit_rate.pct,\
        lts__t_hit_rate.pct,\
        l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
        $<TARGET_FILE:fft_${impl}> ${INPUT_DIR}/1024x1024_matrix.dat
        DEPENDS fft_${impl} create_inputs
    )
    
    # Full analysis target
    add_custom_target(${impl}_analysis
        DEPENDS 
            ${impl}_memtrans
            ${impl}_throughput
            ${impl}_cache
    )
endforeach()

# Add target to analyze all implementations
add_custom_target(analyze_all
    DEPENDS 
        stage0_analysis
        stage1_analysis
        stage2_analysis
        stage3_analysis
        cufft_analysis
)