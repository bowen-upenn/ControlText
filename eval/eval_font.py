import argparse
import albumentations as A
import csv
import huggingface_hub
import numpy as np
import onnxruntime as ort
import os
import yaml
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from scipy.spatial.distance import cosine, euclidean
from PIL import Image
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Tuple
from tqdm import tqdm


# Set device
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG_PATH = huggingface_hub.hf_hub_download(
    repo_id="storia/font-classify-onnx", filename="model_config.yaml"
)
MODEL_PATH = huggingface_hub.hf_hub_download(
    repo_id="storia/font-classify-onnx", filename="model.onnx"
)
MAPPING_PATH = "google_fonts_mapping.tsv"


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def extract_base_image(filename):
    parts = filename.split('_')
    return f"{parts[0]}_{parts[-1]}"


class ResizeWithPad:
    def __init__(self, new_shape: Tuple[int, int], padding_color: Tuple[int] = (255, 255, 255)) -> None:
        self.new_shape = new_shape
        self.padding_color = padding_color

    def __call__(self, image: np.array, **kwargs) -> np.array:
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(self.new_shape)) / max(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])
        image = cv2.resize(image, new_size)
        delta_w = self.new_shape[0] - new_size[0]
        delta_h = self.new_shape[1] - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=self.padding_color,
        )
        return image


class CutMax:
    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size

    def __call__(self, image: np.array, **kwargs) -> np.array:
        if image.shape[0] > self.max_size:
            image = image[: self.max_size, :, :]
        if image.shape[1] > self.max_size:
            image = image[:, : self.max_size, :]
        return image


def load_images_and_generate_probs(image_folder):
    print(f"Begin generating probabilities for images in {image_folder}")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    input_size = config["size"]

    session = ort.InferenceSession(MODEL_PATH)
    transform = A.Compose([
        A.Lambda(image=CutMax(1024)),
        A.Lambda(image=ResizeWithPad((input_size, input_size))),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    arrays = []
    for image_file in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_file)
        image = np.array(Image.open(image_path))
        # extend color dim for grayscale images
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        image = transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)

        logits = session.run(None, {"input": image})[0][0]
        probs = softmax(logits)
        arrays.append([image_file] + list(probs))

    csv_path = f"{image_folder}_probs.csv"
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in arrays:
            writer.writerow(row)

    print(f"Probabilities for images in {image_folder} saved to {csv_path}")
    csv_file = pd.read_csv(csv_path, header=None)
    return csv_file


def filter_top_k(vec, k):
    """
    Returns a copy of vec with only the top k values retained (others zeroed out).
    If k is greater than or equal to the length of vec, returns the original vector.
    """
    vec = np.array(vec)  # Ensure input is an np.array
    if k >= vec.size:
        return vec.copy()
    # Get indices of the top k values
    indices = np.argsort(vec)[-k:]
    # Create a new zeroed vector and fill in the top k values
    filtered = np.zeros_like(vec)
    filtered[indices] = vec[indices]
    return filtered


def calculate_distances(vec1, vec2):
    l2_dist = euclidean(vec1, vec2)
    cosine_dist = cosine(vec1, vec2)
    return l2_dist, cosine_dist


def compute_prob_distances(df_generated, df_gt):
    # Initialize results list to store distances for each image
    results = []
    # Define the k-values to be used (including full which uses the unmodified vector)
    k_values = [5, 20, 50, None]  # None will represent the full vector
    for row_idx, row in tqdm(df_generated.iterrows(), total=df_generated.shape[0]):
        image = row[0]
        base_image = extract_base_image(image)
        generated_probs = row.iloc[1:].values.astype(np.float32)
        gt_probs = df_gt.loc[df_gt[0] == base_image, df_gt.columns[1:]].values.flatten().astype(np.float32)

        # Only compute distances if a matching ground truth exists
        if len(gt_probs) > 0:
            distances = {}
            # Compute distances for each k scenario
            for k in k_values:
                if k is None:  # Full vector
                    gen_vec = generated_probs
                    gt_vec = gt_probs
                    key = 'full'
                else:
                    gen_vec = filter_top_k(generated_probs, k)
                    gt_vec = filter_top_k(gt_probs, k)
                    key = f"top{k}"

                l2_dist, cosine_dist = calculate_distances(gen_vec, gt_vec)
                distances[f"l2_distance_{key}"] = l2_dist
                distances[f"cosine_distance_{key}"] = cosine_dist

            results.append([image, base_image,
                            distances["cosine_distance_top5"],
                            distances["l2_distance_top5"],
                            distances["cosine_distance_top20"],
                            distances["l2_distance_top20"],
                            distances["cosine_distance_top50"],
                            distances["l2_distance_top50"],
                            distances["cosine_distance_full"],
                            distances["l2_distance_full"]])

    # Define column names in the desired order
    columns = ['generated_image', 'gt_image',
               'cosine_distance_top5', 'l2_distance_top5',
               'cosine_distance_top20', 'l2_distance_top20',
               'cosine_distance_top50', 'l2_distance_top50',
               'cosine_distance_full', 'l2_distance_full']

    results_df = pd.DataFrame(results, columns=columns)

    # Print and save results
    print("Results shape:", results_df.shape)
    print(results_df)

    # Compute and print averages for each metric
    for col in results_df.columns[2:]:
        avg_val = results_df[col].mean()
        print(f"Average {col}: {avg_val:.6f}")

    results_df.to_csv('prob_distances.csv', index=False)

    # Save average metrics to a text file
    with open('average_distances.txt', 'w') as f:
        for col in results_df.columns[2:]:
            avg_val = results_df[col].mean()
            f.write(f"Average {col}: {avg_val:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inference with pretrained model from Storia"
    )
    parser.add_argument(
        "--generated_folder",
        type=str,
        required=True,
        help="Path to the folder containing generated images",
    )
    parser.add_argument(
        "--gt_folder",
        type=str,
        required=True,
        help="Path to the folder containing ground truth images",
    )
    args = parser.parse_args()

    print(args.generated_folder + '_probs.csv', args.gt_folder + '_probs.csv')
    if os.path.exists(args.generated_folder + '_probs.csv'):
        print('Loading generated images probabilities from', args.generated_folder + '_probs.csv')
        generated_csv = pd.read_csv(args.generated_folder + '_probs.csv', header=None)
    else:
        print('Generating probabilities for generated images')
        generated_csv = load_images_and_generate_probs(args.generated_folder)

    if os.path.exists(args.gt_folder + '_probs.csv'):
        print('Loading ground truth images probabilities from', args.gt_folder + '_probs.csv')
        gt_csv = pd.read_csv(args.gt_folder + '_probs.csv', header=None)
    else:
        print('Generating probabilities for ground truth images')
        gt_csv = load_images_and_generate_probs(args.gt_folder)

    print("Computing distances between generated and ground truth images")
    compute_prob_distances(generated_csv, gt_csv)


if __name__ == "__main__":
    main()
