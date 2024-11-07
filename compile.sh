#!/bin/bash
echo "Compiling"
nvcc -o stage1 src/fft-stage1-2D.cu 