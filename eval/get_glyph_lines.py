import os
import cv2
import json
import numpy as np

# # JSON path
# wukong_test1k_json_path = '/pool/bwjiang/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/wukong_word/test1k.json'
# laion_test1k_json_path = '/pool/bwjiang/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/laion_word/test1k.json'
#
# # Image path for laion
# laion_image_dir = '/pool/bwjiang/controltext/eval/laion_controltext_ploss2'
# laion_bw_image_dir = '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/laion_ploss2'
#
# # output directory for laion glyphs
# laion_glyph_dir = '/pool/bwjiang/controltext/eval/laion_controltext_ploss2_gly_lines'
# laion_bw_glyph_dir = '/pool/bwjiang/controltext/eval/laion_controltext_ploss2_gly_lines_black_and_white'
#
# # Image path for wukong
# wukong_image_dir = '/pool/bwjiang/controltext/eval/wukong_controltext_ploss2'
# wukong_bw_image_dir = '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/wukong_ploss2'
#
# # output directory for wukong glyphs
# wukong_glyph_dir = '/pool/bwjiang/controltext/eval/wukong_controltext_ploss2_gly_lines'
# wukong_bw_glyph_dir = '/pool/bwjiang/controltext/eval/wukong_controltext_ploss2_gly_lines_black_and_white'


# # JSON path
# wukong_test1k_json_path = '/pool/bwjiang/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/wukong_word/test1k.json'
# laion_test1k_json_path = '/pool/bwjiang/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/laion_word/test1k.json'
#
# # Image path for laion
# laion_image_dir = '/pool/bwjiang/controltext/eval/anytext_eval_imgs/textdiffuser_laion_generated/'
# laion_bw_image_dir =  '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/textdiffuser_laion'
#
# # output directory for laion glyphs
# laion_glyph_dir = '/pool/bwjiang/controltext/eval/textdiffuser_laion_generated_gly_lines'
# laion_bw_glyph_dir = '/pool/bwjiang/controltext/eval/textdiffuser_laion_generated_gly_lines_black_and_white'
#
# # Image path for wukong
# wukong_image_dir = '/pool/bwjiang/controltext/eval/anytext_eval_imgs/textdiffuser_wukong_generated/'
# wukong_bw_image_dir = '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/textdiffuser_wukong'
#
# # output directory for wukong glyphs
# wukong_glyph_dir = '/pool/bwjiang/controltext/eval/textdiffuser_wukong_generated_gly_lines'
# wukong_bw_glyph_dir = '/pool/bwjiang/controltext/eval/textdiffuser_wukong_generated_gly_lines_black_and_white'


# JSON path
wukong_test1k_json_path = '/pool/bwjiang/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/wukong_word/test1k.json'
laion_test1k_json_path = '/pool/bwjiang/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/laion_word/test1k.json'

# Image path for laion
laion_image_dir = '/pool/bwjiang/controltext/eval/anytext_eval_imgs/glyphcontrol_laion_generated/'
laion_bw_image_dir =  '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/glyphcontrol_laion'

# output directory for laion glyphs
laion_glyph_dir = '/pool/bwjiang/controltext/eval/glyphcontrol_laion_generated_gly_lines'
laion_bw_glyph_dir = '/pool/bwjiang/controltext/eval/glyphcontrol_laion_generated_gly_lines_black_and_white'

# Image path for wukong
wukong_image_dir = '/pool/bwjiang/controltext/eval/anytext_eval_imgs/glyphcontrol_wukong_generated/'
wukong_bw_image_dir = '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/glyphcontrol_wukong'

# output directory for wukong glyphs
wukong_glyph_dir = '/pool/bwjiang/controltext/eval/glyphcontrol_wukong_generated_gly_lines'
wukong_bw_glyph_dir = '/pool/bwjiang/controltext/eval/glyphcontrol_wukong_generated_gly_lines_black_and_white'


# # JSON path
# wukong_test1k_json_path = '/pool/bwjiang/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/wukong_word/test1k.json'
# laion_test1k_json_path = '/pool/bwjiang/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/laion_word/test1k.json'
#
# # Image path for laion
# laion_image_dir = '/pool/bwjiang/controltext/eval/anytext_eval_imgs/controlnet_laion_generated/'
# laion_bw_image_dir =  '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/controlnet_laion'
#
# # output directory for laion glyphs
# laion_glyph_dir = '/pool/bwjiang/controltext/eval/controlnet_laion_generated_gly_lines'
# laion_bw_glyph_dir = '/pool/bwjiang/controltext/eval/controlnet_laion_generated_gly_lines_black_and_white'
#
# # Image path for wukong
# wukong_image_dir = '/pool/bwjiang/controltext/eval/anytext_eval_imgs/controlnet_wukong_generated/'
# wukong_bw_image_dir = '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/controlnet_wukong'
#
# # output directory for wukong glyphs
# wukong_glyph_dir = '/pool/bwjiang/controltext/eval/controlnet_wukong_generated_gly_lines'
# wukong_bw_glyph_dir = '/pool/bwjiang/controltext/eval/controlnet_wukong_generated_gly_lines_black_and_white'


def get_glyphs(json_path, image_dir, glyph_dir):
    # Ensure glyph directory exists
    os.makedirs(glyph_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print('Empty JSON file')
        return

    data_list = data.get('data_list', [])

    for entry in data_list:
        img_name_prefix = entry['img_name'].split('.')[0]
        img_annotations = entry.get('annotations', [])

        # Find all images with the given prefix
        matching_images = [f for f in os.listdir(image_dir) if f.startswith(img_name_prefix)]

        if not matching_images:
            print(f'No matching image found for {img_name_prefix}')
            continue

        for img_file in matching_images:
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f'Error: Unable to read image {img_file}')
                continue

            for idx, annotation in enumerate(img_annotations):
                polygon = annotation.get('polygon', [])
                valid = annotation.get('valid', False)
                text_content = annotation.get('text', "")

                if not valid or len(polygon) != 4:
                    continue  # Skip invalid annotations or incorrect polygon format

                # Convert polygon to bounding box
                x, y, w, h = cv2.boundingRect(np.array(polygon, dtype=np.int32))

                # Crop the glyph region
                cropped_img = img[y:y + h, x:x + w]

                # Save cropped glyph image
                if not cropped_img.size:
                    continue
                output_filename = f"{os.path.splitext(img_file)[0]}_{idx}_{text_content}.jpg"
                output_path = os.path.join(glyph_dir, output_filename)
                cv2.imwrite(output_path, cropped_img)
                print(f"Saved cropped glyph: {output_path}")

# get_glyphs(laion_test1k_json_path, laion_image_dir, laion_glyph_dir)
# get_glyphs(wukong_test1k_json_path, wukong_image_dir, wukong_glyph_dir)
get_glyphs(laion_test1k_json_path, laion_bw_image_dir, laion_bw_glyph_dir)
get_glyphs(wukong_test1k_json_path, wukong_bw_image_dir, wukong_bw_glyph_dir)

print("Glyph Processing Done")