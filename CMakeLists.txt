cmake_minimum_required(VERSION 3.18)
project(GPU_FFT CUDA CXX)

# Set CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Find CUDA and OpenMP
find_package(CUDA REQUIRED)
find_package(OpenMP)

# Set compiler flags with optimizations
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --ptxas-options=-v")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Enable CUDA compiler optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --maxrregcount=64")
endif()

# Define directories
set(INPUT_DIR ${CMAKE_SOURCE_DIR}/input)
set(EVAL_DIR ${CMAKE_BINARY_DIR}/eval_results)

# Define matrix sizes (split into small and large)
set(SMALL_SIZES 32 64 128 256 512 1024)
set(LARGE_SIZES 2048 4096 8192 16384)

# Make sure directories exist
file(MAKE_DIRECTORY ${INPUT_DIR})
file(MAKE_DIRECTORY ${EVAL_DIR})

# Add matrix generator executable
add_executable(generate_matrix src/generate_matrix.cpp)
target_link_libraries(generate_matrix PRIVATE stdc++fs)
target_compile_definitions(generate_matrix PRIVATE SINGLE_SIZE_MODE)

# Add FFT executables
add_executable(fft_cpu src/fft-cpu-2D.cc)
add_executable(fft_stage0 src/fft-stage0-2D.cu)
add_executable(fft_stage1 src/fft-stage1-2D.cu)
add_executable(fft_stage2 src/fft-stage2-2D.cu)
add_executable(fft_stage3 src/fft-stage3-2D.cu)
add_executable(fft_cufft src/cufft2d.cu)

# Set common compile definitions for CUDA kernels
foreach(target fft_stage0 fft_stage1 fft_stage2 fft_stage3 fft_cufft)
    target_compile_definitions(${target} PRIVATE 
        CUDA_ARCH=${CMAKE_CUDA_ARCHITECTURES}
        USE_SHARED_MEMORY
        USE_PINNED_MEMORY
    )
endforeach()

# Link libraries and set properties
target_link_libraries(fft_cufft PRIVATE cufft)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fft_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set CUDA properties
foreach(target fft_stage0 fft_stage1 fft_stage2 fft_stage3 fft_cufft)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endforeach()

# Create input generation targets for all sizes
foreach(size ${SMALL_SIZES} ${LARGE_SIZES})
    add_custom_command(
        OUTPUT ${INPUT_DIR}/${size}x${size}_matrix.dat
        COMMAND ${CMAKE_BINARY_DIR}/generate_matrix ${size} ${CMAKE_SOURCE_DIR}
        DEPENDS generate_matrix
        COMMENT "Generating ${size}x${size} matrix if not exists"
    )
    list(APPEND ALL_INPUT_FILES ${INPUT_DIR}/${size}x${size}_matrix.dat)
endforeach()

# Create high-level input generation targets
add_custom_target(create_all_inputs ALL
    DEPENDS ${ALL_INPUT_FILES}
)

# Function to add run targets for a given size
function(add_size_targets size)
    foreach(impl cpu stage0 stage1 stage2 stage3 cufft)
        add_custom_target(run_${impl}_${size}
            COMMAND fft_${impl} ${INPUT_DIR}/${size}x${size}_matrix.dat
            DEPENDS fft_${impl} ${INPUT_DIR}/${size}x${size}_matrix.dat
        )
    endforeach()

    # Add combined target for the size
    add_custom_target(run_all_${size}
        DEPENDS
            run_cpu_${size}
            run_stage0_${size}
            run_stage1_${size}
            run_stage2_${size}
            run_stage3_${size}
            run_cufft_${size}
    )
endfunction()

# Add run targets for all sizes
foreach(size ${SMALL_SIZES} ${LARGE_SIZES})
    add_size_targets(${size})
endforeach()

# Add performance analysis targets with comprehensive metrics
foreach(impl stage0 stage1 stage2 stage3 cufft)
    # Memory transaction analysis
    add_custom_command(
        OUTPUT ${EVAL_DIR}/${impl}_memtrans.csv
        COMMAND ncu
            --set full
            --section MemoryWorkloadAnalysis
            --metrics "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
                      l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
                      l1tex__t_sector_hit_rate.pct,\
                      lts__t_sector_hit_rate.pct,\
                      dram__sectors_per_request_umin.pct,\
                      smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"
            --csv
            --log-file ${EVAL_DIR}/${impl}_memtrans.csv
            --replay-mode kernel
            --replay-count 10
            $<TARGET_FILE:fft_${impl}> ${INPUT_DIR}/64x64_matrix.dat
        DEPENDS fft_${impl} ${INPUT_DIR}/64x64_matrix.dat
    )

    # Data processing for averages
    add_custom_command(
        OUTPUT ${EVAL_DIR}/${impl}_memtrans_avg.csv
        COMMAND python ${CMAKE_SOURCE_DIR}/process_ncu_csv.py ${EVAL_DIR}/${impl}_memtrans.csv ${EVAL_DIR}/${impl}_memtrans_avg.csv
        DEPENDS ${EVAL_DIR}/${impl}_memtrans.csv
        COMMENT "Processing NCU CSV data to compute averages for ${impl}_memtrans"
    )

    add_custom_target(${impl}_memtrans_avg
        DEPENDS ${EVAL_DIR}/${impl}_memtrans_avg.csv
    )

    # Similar commands for throughput, cache, and occupancy analyses
    # Adjusted to use --csv and --log-file instead of --export-format

    # Memory throughput analysis
    add_custom_command(
        OUTPUT ${EVAL_DIR}/${impl}_throughput.csv
        COMMAND ncu
            --set full
            --section MemoryWorkloadAnalysis
            --metrics "dram__bytes_read.sum,\
                      dram__bytes_write.sum,\
                      dram__throughput.avg.pct_of_peak_sustained_elapsed,\
                      l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
                      lts__throughput.avg.pct_of_peak_sustained_elapsed"
            --csv
            --log-file ${EVAL_DIR}/${impl}_throughput.csv
            --replay-mode kernel
            --replay-count 10
            $<TARGET_FILE:fft_${impl}> ${INPUT_DIR}/64x64_matrix.dat
        DEPENDS fft_${impl} ${INPUT_DIR}/64x64_matrix.dat
    )

    add_custom_command(
        OUTPUT ${EVAL_DIR}/${impl}_throughput_avg.csv
        COMMAND python ${CMAKE_SOURCE_DIR}/process_ncu_csv.py ${EVAL_DIR}/${impl}_throughput.csv ${EVAL_DIR}/${impl}_throughput_avg.csv
        DEPENDS ${EVAL_DIR}/${impl}_throughput.csv
        COMMENT "Processing NCU CSV data to compute averages for ${impl}_throughput"
    )

    add_custom_target(${impl}_throughput_avg
        DEPENDS ${EVAL_DIR}/${impl}_throughput_avg.csv
    )

    # Cache and compute efficiency analysis
    add_custom_command(
        OUTPUT ${EVAL_DIR}/${impl}_cache.csv
        COMMAND ncu
            --set full
            --section SpeedOfLight_RooflineChart
            --metrics "l1tex__t_hit_rate.pct,\
                      lts__t_hit_rate.pct,\
                      sm__warps_active.avg.pct_of_peak_sustained_active,\
                      sm__sass_thread_inst_executed_per_launched_thread.ratio,\
                      l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
            --csv
            --log-file ${EVAL_DIR}/${impl}_cache.csv
            --replay-mode kernel
            --replay-count 10
            $<TARGET_FILE:fft_${impl}> ${INPUT_DIR}/64x64_matrix.dat
        DEPENDS fft_${impl} ${INPUT_DIR}/64x64_matrix.dat
    )

    add_custom_command(
        OUTPUT ${EVAL_DIR}/${impl}_cache_avg.csv
        COMMAND python ${CMAKE_SOURCE_DIR}/process_ncu_csv.py ${EVAL_DIR}/${impl}_cache.csv ${EVAL_DIR}/${impl}_cache_avg.csv
        DEPENDS ${EVAL_DIR}/${impl}_cache.csv
        COMMENT "Processing NCU CSV data to compute averages for ${impl}_cache"
    )

    add_custom_target(${impl}_cache_avg
        DEPENDS ${EVAL_DIR}/${impl}_cache_avg.csv
    )

    # Occupancy and launch statistics
    add_custom_command(
        OUTPUT ${EVAL_DIR}/${impl}_occupancy.csv
        COMMAND ncu
            --set full
            --section LaunchStats
            --metrics "sm__threads_launched_per_multiprocessor.avg,\
                      sm__warps_active.avg.pct_of_peak_sustained_active,\
                      sm__maximum_warps_per_active_cycle_pct,\
                      l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"
            --csv
            --log-file ${EVAL_DIR}/${impl}_occupancy.csv
            --replay-mode kernel
            --replay-count 10
            $<TARGET_FILE:fft_${impl}> ${INPUT_DIR}/64x64_matrix.dat
        DEPENDS fft_${impl} ${INPUT_DIR}/64x64_matrix.dat
    )

    add_custom_command(
        OUTPUT ${EVAL_DIR}/${impl}_occupancy_avg.csv
        COMMAND python ${CMAKE_SOURCE_DIR}/process_ncu_csv.py ${EVAL_DIR}/${impl}_occupancy.csv ${EVAL_DIR}/${impl}_occupancy_avg.csv
        DEPENDS ${EVAL_DIR}/${impl}_occupancy.csv
        COMMENT "Processing NCU CSV data to compute averages for ${impl}_occupancy"
    )

    add_custom_target(${impl}_occupancy_avg
        DEPENDS ${EVAL_DIR}/${impl}_occupancy_avg.csv
    )

    # Full analysis target
    add_custom_target(${impl}_analysis
        DEPENDS 
            ${impl}_memtrans_avg
            ${impl}_throughput_avg
            ${impl}_cache_avg
            ${impl}_occupancy_avg
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

# Optional: Add a target to run all sizes
add_custom_target(run_all_sizes)
foreach(size ${SMALL_SIZES} ${LARGE_SIZES})
    add_dependencies(run_all_sizes run_all_${size})
endforeach()

# Add benchmark targets for comparison
add_custom_target(benchmark_small
    DEPENDS create_all_inputs
)
foreach(size ${SMALL_SIZES})
    add_dependencies(benchmark_small run_all_${size})
endforeach()

add_custom_target(benchmark_large
    DEPENDS create_all_inputs
)
foreach(size ${LARGE_SIZES})
    add_dependencies(benchmark_large run_all_${size})
endforeach()

# Add comprehensive report target
add_custom_target(generate_report
    COMMAND ${CMAKE_COMMAND} -E echo "Generating performance report..."
    COMMAND ${CMAKE_COMMAND} -E cat ${EVAL_DIR}/*_avg.csv > ${EVAL_DIR}/comprehensive_report.csv
    DEPENDS analyze_all
)

# Ensure that the evaluation directory exists
file(MAKE_DIRECTORY ${EVAL_DIR})
