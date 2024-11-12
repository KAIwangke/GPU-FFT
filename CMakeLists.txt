cmake_minimum_required(VERSION 3.18)
project(GPU_FFT CUDA CXX)

# Set CUDA architectures if not already defined
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenMP)

# Set compiler flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Define directories
set(INPUT_DIR ${CMAKE_SOURCE_DIR}/input)
set(EVAL_DIR ${CMAKE_BINARY_DIR}/eval_results)

# Add executables
add_executable(fft_cpu src/fft-cpu-2D.cc)
add_executable(fft_stage0 src/fft-stage0-2D.cu)
add_executable(fft_stage1 src/fft-stage1-2D.cu)
add_executable(fft_stage2 src/fft-stage2-2D.cu)
add_executable(fft_stage3 src/fft-stage3-2D.cu)
add_executable(fft_stage4 src/fft-stage4-2D-multi.cu)
add_executable(fft_cufft src/cufft2d.cu)

# Set CUDA properties for each target
foreach(target fft_stage0 fft_stage1 fft_stage2 fft_stage3 fft_stage4 fft_cufft)
    set_target_properties(${target} PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()

# Link libraries
target_link_libraries(fft_stage4 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(fft_cufft PRIVATE cufft)

# Function to add evaluation targets for a specific implementation
function(add_perf_analysis TARGET_NAME)
    set(EVAL_OUTPUT_DIR ${EVAL_DIR}/${TARGET_NAME})
    file(MAKE_DIRECTORY ${EVAL_OUTPUT_DIR})

    add_custom_target(${TARGET_NAME}_memtrans
        COMMAND ncu --metrics "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct" 
        --csv --page raw
        $<TARGET_FILE:${TARGET_NAME}> ${INPUT_DIR}/1000.dat > ${EVAL_OUTPUT_DIR}/memtrans.csv
        DEPENDS ${TARGET_NAME}
    )

    add_custom_target(${TARGET_NAME}_throughput
        COMMAND ncu --metrics "dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed"
        --csv --page raw
        $<TARGET_FILE:${TARGET_NAME}> ${INPUT_DIR}/1000.dat > ${EVAL_OUTPUT_DIR}/throughput.csv
        DEPENDS ${TARGET_NAME}
    )

    add_custom_target(${TARGET_NAME}_cache
        COMMAND ncu --metrics "l1tex__t_hit_rate.pct,lts__t_hit_rate.pct,l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
        --csv --page raw
        $<TARGET_FILE:${TARGET_NAME}> ${INPUT_DIR}/1000.dat > ${EVAL_OUTPUT_DIR}/cache.csv
        DEPENDS ${TARGET_NAME}
    )

    add_custom_target(${TARGET_NAME}_pattern
        COMMAND ncu --metrics "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio"
        --csv --page raw
        $<TARGET_FILE:${TARGET_NAME}> ${INPUT_DIR}/1000.dat > ${EVAL_OUTPUT_DIR}/pattern.csv
        DEPENDS ${TARGET_NAME}
    )

    add_custom_target(${TARGET_NAME}_workload
        COMMAND ncu --section MemoryWorkloadAnalysis --section-folder ${EVAL_OUTPUT_DIR}/workload
        $<TARGET_FILE:${TARGET_NAME}> ${INPUT_DIR}/1000.dat
        DEPENDS ${TARGET_NAME}
    )

    # Add full analysis target
    add_custom_target(${TARGET_NAME}_analysis
        DEPENDS 
            ${TARGET_NAME}_memtrans
            ${TARGET_NAME}_throughput
            ${TARGET_NAME}_cache
            ${TARGET_NAME}_pattern
            ${TARGET_NAME}_workload
    )
endfunction()

# Add basic run targets
foreach(impl cpu stage0 stage1 stage2 stage3 stage4 cufft)
    add_custom_target(run_${impl}_64
        COMMAND fft_${impl} ${INPUT_DIR}/64.dat
        DEPENDS fft_${impl}
    )
    add_custom_target(run_${impl}_256
        COMMAND fft_${impl} ${INPUT_DIR}/256x256_matrix.dat
        DEPENDS fft_${impl}
    )
    add_custom_target(run_${impl}_1000
        COMMAND fft_${impl} ${INPUT_DIR}/1000.dat
        DEPENDS fft_${impl}
    )
endforeach()

# Add performance analysis targets for CUDA implementations
foreach(impl stage0 stage1 stage2 stage3 stage4 cufft)
    add_perf_analysis(fft_${impl})
endforeach()

# Add comprehensive targets
add_custom_target(run_all_64
    DEPENDS run_cpu_64 run_stage0_64 run_stage1_64 run_stage2_64 run_stage3_64 run_stage4_64 run_cufft_64
)

add_custom_target(run_all_256
    DEPENDS run_cpu_256 run_stage0_256 run_stage1_256 run_stage2_256 run_stage3_256 run_stage4_256 run_cufft_256
)

add_custom_target(run_all_1000
    DEPENDS run_cpu_1000 run_stage0_1000 run_stage1_1000 run_stage2_1000 run_stage3_1000 run_stage4_1000 run_cufft_1000
)

add_custom_target(analyze_all
    DEPENDS 
        fft_stage0_analysis fft_stage1_analysis fft_stage2_analysis 
        fft_stage3_analysis fft_stage4_analysis fft_cufft_analysis
)