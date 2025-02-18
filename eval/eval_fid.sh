#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -m pytorch_fid \
    /tmp/datasets/AnyWord-3M/AnyText-Benchmark/FID/laion-40k \
    ./eval/laion_controltext