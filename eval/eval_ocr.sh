#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python eval/eval_dgocr.py \
        --img_dir ./eval/laion_controltext \
        --json_path /tmp/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/laion_word/test1k.json \
        --glyph_path ./Rethinking-Text-Segmentation/log/images/output/anytext_benchmark/laion_word