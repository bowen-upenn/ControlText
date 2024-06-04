#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python eval/eval_dgocr.py \
        --img_dir /tmp/datasets/AnyWord-3M/anytext_eval_imgs/anytext_v1.1_laion_generated \
        --input_json /tmp/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/laion_word/test1k.json