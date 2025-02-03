#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python eval/eval_dgocr.py \
        --img_dir ./eval/wukong_anytext \
        --json_path /tmp/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/wukong_word/test1k.json \
        --glyph_path ./Rethinking-Text-Segmentation/log/images/output/anytext_benchmark/wukong_word