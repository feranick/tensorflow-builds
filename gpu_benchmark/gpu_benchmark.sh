#!/bin/bash

echo
echo "Tensor Core: False, fp32"
python3 gpu_benchmark_tf-false.py

echo
echo "Tensor Core: True, fp32"
python3 gpu_benchmark_tf-true.py

echo
echo "Tensor Core: True, fp16"
python3 gpu_benchmark_tf-true_fp16.py
