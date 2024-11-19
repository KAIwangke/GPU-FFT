#!/bin/bash

MATRIX_SIZE=1024
RUNS=5

# First ensure input directory exists
mkdir -p ../input

# Array of GPU implementations to test
IMPLEMENTATIONS=("stage0" "stage1" "stage2" "stage3" "cufft")

echo "Testing GPU FFT implementations on ${MATRIX_SIZE}x${MATRIX_SIZE} matrix ($RUNS runs each)"
echo "----------------------------------------"

# First verify input file exists
if [ ! -f "../input/${MATRIX_SIZE}x${MATRIX_SIZE}_matrix.dat" ]; then
    echo "Generating input matrix..."
    ./generate_matrix $MATRIX_SIZE ..
fi

for impl in "${IMPLEMENTATIONS[@]}"; do
    echo -n "$impl: "
    
    # Run and collect times
    total=0
    valid_runs=0
    
    for ((i=1; i<=RUNS; i++)); do
        # Note: Changed to look for "Total execution time:" and "ms"
        time=$(./fft_${impl} "../input/${MATRIX_SIZE}x${MATRIX_SIZE}_matrix.dat" 2>&1 | grep "Total execution time:" | awk '{print $4}')
        
        if [ ! -z "$time" ]; then
            total=$(echo "$total + $time" | bc -l)
            valid_runs=$((valid_runs + 1))
        fi
    done
    
    # Calculate and print average if we have valid runs
    if [ $valid_runs -gt 0 ]; then
        average=$(echo "scale=6; $total / $valid_runs" | bc -l)
        echo "$average ms"
    else
        echo "Failed to get valid timing"
    fi
done