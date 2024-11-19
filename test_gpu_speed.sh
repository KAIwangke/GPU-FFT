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
    echo "Running $impl implementation:"
    
    # Run implementation multiple times
    for ((i=1; i<=RUNS; i++)); do
        echo "Run $i:"
        ./fft_${impl} "../input/${MATRIX_SIZE}x${MATRIX_SIZE}_matrix.dat"
        echo "----------------------------------------"
    done
done
