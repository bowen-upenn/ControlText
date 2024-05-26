import os
import json
import numpy as np
import cv2
import random
import string
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps, ImageColor
from typing import List, Any
import math
import torch
import torch.nn.functional as F
import scipy
from scipy.ndimage import map_coordinates


# Recover from random curvatures
def find_min_max_coordinates(image):
    """Find the non-black pixel with min and max x coordinates."""
    image = np.array(image)
    y_indices, x_indices = np.nonzero(np.any(image > 0, axis=-1))
    min_x = np.min(x_indices)
    max_x = np.max(x_indices)
    # Assuming straight line can be approximated by the average y-values at these x-positions
    min_y = int(np.mean(y_indices[x_indices == min_x]))
    max_y = int(np.mean(y_indices[x_indices == max_x]))
    return (min_x, min_y), (max_x, max_y)


def calculate_line_equation(p1, p2):
    """Calculate the coefficients A, B, and C of the line equation Ax + By + C = 0."""
    (x1, y1), (x2, y2) = p1, p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C


def calculate_line_slope(p1, p2):
    """Calculate the slope of the line passing through two points."""
    (x1, y1), (x2, y2) = p1, p2
    return (y2 - y1) / (x2 - x1)


def check_if_point_is_above_the_line(p, p1, p2):
    """Check if a point is above the line passing through two points."""
    (x, y) = p
    A, B, C = calculate_line_equation(p1, p2)
    return A * x + B * y + C > 0


def correct_curvature(curve_text_img, curved_line_img, image_size, p1, p2):
    """Correct the curvature by aligning curved line pixels to a straight reference line."""
    width, height = image_size
    A, B, C = calculate_line_equation(p1, p2)
    corrected_img_array = np.zeros_like(curve_text_img)

    for x in range(width):
        # Find all non-black pixels in the column
        y_curved = np.nonzero(np.any(curved_line_img[:, x] > 0, axis=-1))[0]
        if y_curved.size > 0:
            # Calculate perpendicular projection of each y onto the line
            y_projected = - (A * x + C) / B
            y_shift = int(y_projected - np.mean(y_curved))  # Calculate shift needed

            # Apply shift to all pixels in the column
            for y in range(height):
                new_y = y + y_shift
                if 0 <= new_y < height:
                    corrected_img_array[new_y, x] = curve_text_img[y, x]

    return corrected_img_array


# Recover from random perspective transformations
def detect_gaussian_corners(image, num_lines=1, curr_path=None):
    # Step 1: Load image and convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply GaussianBlur to smooth the image (optional, adjust parameters as needed)
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)

    # Step 3: Apply a threshold to isolate dots
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)  # Adjust threshold value as necessary

    # Step 4: Detect blobs or contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contour_img = np.zeros_like(gray_image)  # Create a black image the same size as the original
    # cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 2)  # Draw white contours
    # cv2.imwrite(curr_path + "_contour.png", contour_img)

    dots = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:   # the area of the contour.
            cX = int(M["m10"] / M["m00"])   # centroid coordinates
            cY = int(M["m01"] / M["m00"])
            dots.append((cX, cY))

    # return dots
    if num_lines == 1:
        if len(dots) == 4:
            return dots
        else:
            # print("Four corners must be detected, detected: {}".format(len(dots)))
            return None
    else:
        return dots


def order_corners(curved_line_img, corners):
    # Find the curved line
    midline_start, midline_end = find_min_max_coordinates(curved_line_img)
    above_line = np.array([check_if_point_is_above_the_line(corner, midline_start, midline_end) for corner in corners])

    # top left is the corner point above the midline with the smaller x coordinate
    corners = np.array(corners)
    top_left = corners[above_line][np.argmin(corners[above_line][:, 0])]
    top_right = corners[above_line][np.argmax(corners[above_line][:, 0])]
    bottom_left = corners[~above_line][np.argmin(corners[~above_line][:, 0])]
    bottom_right = corners[~above_line][np.argmax(corners[~above_line][:, 0])]
    # print('top_left', top_left, 'top_right', top_right, 'bottom_left', bottom_left, 'bottom_right', bottom_right)

    # Assert that all indices are unique
    assert np.sum(above_line) == 2 and np.sum(top_left - top_right) != 0 and np.sum(bottom_left - bottom_right) != 0, "Corner points must be unique"
    ordered_corners = [top_left, top_right, bottom_left, bottom_right]

    return ordered_corners


def create_rectangle(image_size, text_width, text_height):
    W, H = image_size
    center_x, center_y = W // 2, H // 2
    top_left = (center_x - text_width // 2, center_y - text_height // 2)
    top_right = (center_x + text_width // 2, center_y - text_height // 2)
    bottom_left = (center_x - text_width // 2, center_y + text_height // 2)
    bottom_right = (center_x + text_width // 2, center_y + text_height // 2)

    # img = Image.new('RGB', image_size, "black")
    # draw = ImageDraw.Draw(img)
    # draw.polygon([top_left, top_right, bottom_right, bottom_left], outline="white")
    # img.save("toy_examples/recovered/target_rectangle.png")

    return [top_left, top_right, bottom_left, bottom_right]


def perform_perspective_transform(image, image_size, src_points, dst_points):
    # Convert corner lists to NumPy arrays
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    # Apply the perspective transformation to the image (for demonstration, creating a blank image)
    transformed_img = cv2.warpPerspective(image, matrix, (image_size[0], image_size[1]))
    return transformed_img


if __name__ == "__main__":
    output_dir = "toy_examples"
    midline_dir = os.path.join(output_dir, "midline")
    corners_dir = os.path.join(output_dir, "corners")
    target_dir = os.path.join(output_dir, "target")
    target_curved_dir = os.path.join(output_dir, "target_curved")