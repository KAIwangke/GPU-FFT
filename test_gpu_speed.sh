#!/bin/bash

MATRIX_SIZE=1024
RUNS=5

# Check if input file exists
INPUT_FILE="../input/${MATRIX_SIZE}x${MATRIX_SIZE}_matrix.dat"
if [ ! -f "$INPUT_FILE" ]; then
    echo "Generating input matrix..."
    ./generate_matrix $MATRIX_SIZE ..
fi

# Array of GPU implementations to test
IMPLEMENTATIONS=("stage0" "stage1" "stage2" "stage3" "cufft")

echo "Testing GPU FFT implementations on ${MATRIX_SIZE}x${MATRIX_SIZE} matrix ($RUNS runs each)"
echo "----------------------------------------"

for impl in "${IMPLEMENTATIONS[@]}"; do
    echo -n "$impl: "
    
    # Run and collect times
    total=0
    for ((i=1; i<=RUNS; i++)); do
        time=$(./fft_${impl} ${INPUT_FILE} 2>&1 | grep "Execution time:" | awk '{print $3}')
        if [ ! -z "$time" ]; then
            total=$(echo "$total + $time" | bc -l)
        else
            echo "Error: No timing output from fft_${impl}"
            continue
        fi
    done
    
    # Calculate and print average
    if [ "$total" != "0" ]; then
        average=$(echo "scale=6; $total / $RUNS" | bc -l)
        echo "$average seconds"
    else
        echo "Failed to get timing"
    fi
done