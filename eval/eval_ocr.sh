#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
python eval/eval_dgocr.py \
        --img_dir ./eval/wukong_ablation_gly_lines \
        --json_path /tmp/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/wukong_word/test1k.json \
        --glyph_path ./Rethinking-Text-Segmentation/log/images/output/anytext_benchmark/wukong_word

# #!/bin/bash
# export CUDA_VISIBLE_DEVICES=7
# python eval/eval_dgocr.py \
#         --img_dir ./eval/textdiffuser_laion_generated_gly_lines \
#         --json_path /tmp/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/laion_word/test1k.json \
#         --glyph_path ./Rethinking-Text-Segmentation/log/images/output/anytext_benchmark/laion_word

# #!/bin/bash
# export CUDA_VISIBLE_DEVICES=7
# python eval/eval_dgocr.py \
#         --img_dir ./eval/glyphcontrol_laion_generated_gly_lines \
#         --json_path /tmp/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/laion_word/test1k.json \
#         --glyph_path ./Rethinking-Text-Segmentation/log/images/output/anytext_benchmark/laion_word

# #!/bin/bash
# export CUDA_VISIBLE_DEVICES=7
# python eval/eval_dgocr.py \
#         --img_dir ./eval/controlnet_wukong_generated_gly_lines \
#         --json_path /tmp/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/wukong_word/test1k.json \
#         --glyph_path ./Rethinking-Text-Segmentation/log/images/output/anytext_benchmark/wukong_word