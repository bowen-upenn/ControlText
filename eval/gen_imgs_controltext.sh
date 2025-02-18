#!/bin/bash
python eval/anytext_singleGPU.py \
        --ckpt_path ./models/lightning_logs/version_9/checkpoints/last.ckpt \
        --json_path /tmp/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/laion_word/test1k.json \
        --output_dir ./eval/laion_controltext \
        --glyph_path ./Rethinking-Text-Segmentation/log/images/output/anytext_benchmark/laion_word