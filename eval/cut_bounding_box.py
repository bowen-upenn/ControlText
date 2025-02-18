import os
import json
from PIL import Image

def crop_and_save_images_gt(json_data, directory, output_directory, invalid_gly_lines):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    print(output_directory)

    # Process each image entry in the JSON data
    for image_data in json_data["data_list"]:
        img_name = image_data["img_name"]
        annotations = image_data["annotations"]

        # Try to find the image in the specified directories
        img_path = None
        potential_path = os.path.join(directory, img_name)
        print("potential_path: ", potential_path)
        if os.path.exists(potential_path):
            img_path = potential_path

        # If the image was not found, skip to the next entry
        if not img_path:
            print(f"Image not found: {img_name}")
            continue

        # Open the image
        with Image.open(img_path) as img:
            for annotation in annotations:
                polygon = annotation["polygon"]
                text = annotation["text"]
                if img_name in invalid_gly_lines:
                    invalid_texts = [bad_text["text"] for bad_text in invalid_gly_lines[img_name]]
                    if text in invalid_texts:
                        print(f"Skipping invalid text '{text}' in image '{img_name}'")
                        continue

                # Calculate the bounding box from the polygon
                x_coords = [point[0] for point in polygon]
                y_coords = [point[1] for point in polygon]
                bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

                # Crop the image
                cropped_img = img.crop(bbox)

                # Create a new file name
                sanitized_text = text.replace("/", "")
                new_file_name = f"{os.path.splitext(img_name)[0]}_{sanitized_text}.jpg"
                new_file_path = os.path.join(output_directory, new_file_name)

                # Save the cropped image
                cropped_img.save(new_file_path, "JPEG")
                print(f"Saved cropped image: {new_file_path}")

# Define the function to crop and save bounding boxes
def crop_and_save_images(json_data, directory, output_suffix, invalid_gly_lines):
    # Create output directory
    output_directory = os.path.join(directory, output_suffix)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    print(output_directory)

    # Process each image entry in the JSON data
    for image_data in json_data["data_list"]:
        img_name = image_data["img_name"]
        annotations = image_data["annotations"]

        # Try to find the image in the specified directories
        img_path = None
        for i in range(4):
            final_img_name = f"{os.path.splitext(img_name)[0]}_{str(i)}.jpg"
            potential_path = os.path.join(directory, final_img_name)
            print("potential_path: ", potential_path)
            if os.path.exists(potential_path):
                img_path = potential_path

            # If the image was not found, skip to the next entry
            if not img_path:
                print(f"Image not found: {final_img_name}")
                continue

            # Open the image
            with Image.open(img_path) as img:
                for annotation in annotations:
                    polygon = annotation["polygon"]
                    text = annotation["text"]
                    if img_name in invalid_gly_lines:
                        invalid_texts = [bad_text["text"] for bad_text in invalid_gly_lines[img_name]]
                        if text in invalid_texts:
                            print(f"Skipping invalid text '{text}' in image '{final_img_name}'")
                            continue

                    # Calculate the bounding box from the polygon
                    x_coords = [point[0] for point in polygon]
                    y_coords = [point[1] for point in polygon]
                    bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

                    # Crop the image
                    cropped_img = img.crop(bbox)

                    # Create a new file name
                    sanitized_text = text.replace("/", "")
                    new_file_name = f"{os.path.splitext(final_img_name)[0]}_{sanitized_text}.jpg"
                    new_file_path = os.path.join(output_directory, new_file_name)

                    # Save the cropped image
                    cropped_img.save(new_file_path, "JPEG")
                    print(f"Saved cropped image: {new_file_path}")

# Load JSON data
laion_test1k = "/pool/bwjiang/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/laion_word/test1k.json"  # Replace with your JSON file path
wukong_test1k = "/pool/bwjiang/datasets/AnyWord-3M/AnyText-Benchmark/benchmark/wukong_word/test1k.json"
output_suffix = "gly_lines"  # Replace with your desired output directory
invalid_gly_lines_path = "/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/ocr_verified/invalid_gly_lines_test.json"
laion_directories = [
    '/pool/bwjiang/controltext/eval/laion_ablation',
    '/pool/bwjiang/controltext/eval/laion_anytext',
    '/pool/bwjiang/controltext/eval/laion_controltext'
]
laion_directories_segmented = [
    '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/laion_ablation',
    '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/laion_anytext',
    '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/laion_controltext'
]
wukong_directories = [
    '/pool/bwjiang/controltext/eval/wukong_ablation',
    '/pool/bwjiang/controltext/eval/wukong_anytext',
    '/pool/bwjiang/controltext/eval/wukong_controltext'
]
wukong_directories_segmented = [
    '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/wukong_ablation',
    '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/wukong_anytext',
    '/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/wukong_controltext'
]

with open(invalid_gly_lines_path, "r") as file:
    invalid_gly_lines = json.load(file)
with open(laion_test1k, "r") as file:
    laion_json = json.load(file)
with open(wukong_test1k, "r") as file:
    wukong_json = json.load(file)

updated_invalid_gly_lines = {
    os.path.basename(key): value for key, value in invalid_gly_lines.items()
}

# for directory in laion_directories:
#     crop_and_save_images(laion_json, directory, output_suffix, updated_invalid_gly_lines)
#     print("finish laion run")
# 
# for directory in wukong_directories:
#     crop_and_save_images(wukong_json, directory, output_suffix, updated_invalid_gly_lines)
#     print("finish wukong run")
# 
# for directory in laion_directories_segmented:
#     crop_and_save_images(laion_json, directory, output_suffix + "_black_and_white", updated_invalid_gly_lines)
#     print("finish black and white laion")
# 
# for directory in wukong_directories_segmented:
#     crop_and_save_images(wukong_json, directory, output_suffix + "_black_and_white", updated_invalid_gly_lines)
#     print("finish black and white wukong")

output_directory = "/pool/bwjiang/controltext/eval/laion_gly_lines_gt"
directory = "/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/anytext_benchmark/laion_word"
crop_and_save_images_gt(laion_json, directory, output_directory, updated_invalid_gly_lines)
print("finish laion input")

output_directory = "/pool/bwjiang/controltext/eval/wukong_gly_lines_gt"
directory = "/pool/bwjiang/controltext/Rethinking-Text-Segmentation/log/images/output/anytext_benchmark/wukong_word"
crop_and_save_images_gt(wukong_json, directory, output_directory, updated_invalid_gly_lines)
print("finish wukong input")

# Process the images and save cropped regions
# crop_and_save_images(json_data, img_directories, output_directory)
