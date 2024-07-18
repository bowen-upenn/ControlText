import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import math
import cv2

from generate_text_transformation_pairs import gaussian
from restore_from_transformations import find_min_max_coordinates


class SyntheticDataset(Dataset):
    def __init__(self, images_dir, targets_curved_dir, target_corners_dir, target_midlines_dir, image_size, step):
        if step == 'extract':
            self.sources_dir = images_dir
            self.targets_dir = targets_curved_dir
        elif step == 'rectify':
            self.sources_dir = targets_curved_dir
            self.target_corners_dir = target_corners_dir
            self.target_midlines_dir = target_midlines_dir
        else:
            raise ValueError('Invalid step. Please choose between "extract" and "rectify"')

        self.images_dir = images_dir
        self.images = os.listdir(images_dir)
        self.step = step
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             transforms.Resize((image_size, image_size), antialias=True)])
        self.transform_grayscale = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((image_size, image_size), antialias=True)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        if self.step == 'extract':
            source_path = os.path.join(self.sources_dir, img_name)
            target_path = os.path.join(self.targets_dir, img_name)

            source = Image.open(source_path).convert('RGB')
            target = Image.open(target_path).convert('L')

            if self.transform is not None:
                source = self.transform(source)
                target = self.transform_grayscale(target)

            # Process the target image to extract the binary mask of the texts
            target = (target != 0).squeeze(0).float()
            return source, target
        else:
            source_path = os.path.join(self.sources_dir, img_name)
            target_corners_path = os.path.join(self.target_corners_dir, img_name)
            target_midlines_path = os.path.join(self.target_midlines_dir, img_name)

            source = Image.open(source_path).convert('RGB')
            target_corners = Image.open(target_corners_path).convert('L')
            target_midlines = Image.open(target_midlines_path)
            target_midline_endpoints = self.draw_end_points(source.size, target_midlines)
            target_midlines = self.convert_line_to_gaussian(target_midlines)
            target_midlines = target_midlines.convert('L')

            if self.transform is not None:
                source = self.transform(source)
                target_corners = self.transform_grayscale(target_corners)
                target_midlines = self.transform_grayscale(target_midlines)
                target_midline_endpoints = self.transform_grayscale(target_midline_endpoints)

            # Process the target image to extract the binary mask of the texts
            source = (source != 0).any(axis=0).float().unsqueeze(0).repeat(3, 1, 1)
            target_corners = target_corners.squeeze(0).float()
            target_midlines = target_midlines.squeeze(0).float()
            target_midline_endpoints = target_midline_endpoints.squeeze(0).float()

            return source, target_corners, target_midlines, target_midline_endpoints


    def draw_end_points(self, image_size, target_midlines):
        midline_start, midline_end = find_min_max_coordinates(target_midlines)

        # Create an image for the rectangle
        end_points_img = Image.new('L', image_size, 'black')
        rect_draw = ImageDraw.Draw(end_points_img)
        circle_diameter = 20  # You can adjust the size of the circles here
        sigma = circle_diameter / 3  # Standard deviation for Gaussian blur

        ends = [midline_start, midline_end]

        # Draw Gaussian dots at each end
        for end in ends:
            for dx in range(-circle_diameter, circle_diameter):
                for dy in range(-circle_diameter, circle_diameter):
                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    if dist <= circle_diameter:
                        intensity = int(255 * gaussian(dist, 0, sigma))
                        x = end[0] + dx
                        y = end[1] + dy
                        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:  # Check bounds
                            rect_draw.point((x, y), fill=intensity)

        return np.array(end_points_img)


    def convert_line_to_gaussian(self, target_midline):
        target_midline = np.array(target_midline)
        blurred_midline = cv2.GaussianBlur(target_midline, (15, 15), 0)
        # blurred_midline = blurred_midline + target_midline
        # blurred_midline[blurred_midline > 255] = 255
        blurred_midline = Image.fromarray(blurred_midline)
        # blurred_midline = Image.fromarray(blurred_midline.astype(np.uint8))
        return blurred_midline