#!/bin/bash

MATRIX_SIZE=1024
RUNS=5

# Array of GPU implementations to test
IMPLEMENTATIONS=("stage0" "stage1" "stage2" "stage3" "cufft")

echo "Testing GPU FFT implementations on ${MATRIX_SIZE}x${MATRIX_SIZE} matrix ($RUNS runs each)"
echo "----------------------------------------"

for impl in "${IMPLEMENTATIONS[@]}"; do
    echo -n "$impl: "
    
    # Run and collect times
    times=()
    for ((i=1; i<=RUNS; i++)); do
        time=$(./fft_$impl input/${MATRIX_SIZE}x${MATRIX_SIZE}_matrix.dat 2>&1 | grep "Execution time:" | awk '{print $3}')
        times+=($time)
    done
    
    # Calculate average
    total=0
    for t in "${times[@]}"; do
        total=$(echo "$total + $t" | bc -l)
    done
    average=$(echo "scale=6; $total / $RUNS" | bc -l)
    
    echo "$average seconds"
done