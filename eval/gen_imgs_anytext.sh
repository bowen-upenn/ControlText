#!/bin/bash
python eval/anytext_multiGPUs.py \
        --model_path models/anytext_v1.1.ckpt \
        --json_path /tmp/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/laion_word/test1k.json \
        --output_dir ./anytext_laion_generated \
        --gpus 6,7
