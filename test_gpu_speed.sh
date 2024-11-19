#!/bin/bash

MATRIX_SIZE=1024
RUNS=5

# Check for input matrix
if [ ! -f "../input/${MATRIX_SIZE}x${MATRIX_SIZE}_matrix.dat" ]; then
    echo "Generating input matrix..."
    ./generate_matrix $MATRIX_SIZE ..
else
    echo "Using existing input matrix"
fi

echo "Testing GPU FFT implementations on ${MATRIX_SIZE}x${MATRIX_SIZE} matrix ($RUNS runs each)"
echo "----------------------------------------"

IMPLEMENTATIONS=("stage0" "stage1" "stage2" "stage3" "cufft")

for impl in "${IMPLEMENTATIONS[@]}"; do
    echo -n "$impl: "
    total=0
    valid_runs=0

    # Run implementation multiple times
    for ((i=1; i<=RUNS; i++)); do
        output=$(./fft_${impl} "../input/${MATRIX_SIZE}x${MATRIX_SIZE}_matrix.dat" 2>&1)
        # Extract the numeric execution time from the output
        time=$(echo "$output" | grep "Total execution time:" | awk '{print $(NF-1)}')

        if [ ! -z "$time" ]; then
            total=$(echo "$total + $time" | bc)
            valid_runs=$((valid_runs + 1))
        fi
    done

    if [ $valid_runs -gt 0 ]; then
        average=$(echo "scale=4; $total / $valid_runs" | bc)
        echo "$average ms"
    else
        echo "Failed to get timing"
    fi
done
