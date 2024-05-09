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

from restore_from_transformations import *



def load_fonts(fonts_dir, chinese=False, popular_fonts=None, popular_font_weight=0.7):
    """
    Load font paths from a directory, giving higher probability to popular fonts if specified.
    """
    if chinese:  # some latin fonts does not support chinese characters
        fonts_dir = os.path.join(fonts_dir, 'chinese')
    all_fonts = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith('.ttf') or f.endswith('.otf')]

    if popular_fonts:
        popular_fonts = [f for f in all_fonts if any(popular_font in f for popular_font in popular_fonts)]
        weights = [popular_font_weight / len(popular_fonts)] * len(popular_fonts) + \
                  [(1 - popular_font_weight) / (len(all_fonts) - len(popular_fonts))] * (len(all_fonts) - len(popular_fonts))
        return random.choices(all_fonts, weights, k=1)[0]
    else:
        return random.choice(all_fonts)


def get_text_dimensions(draw, text, font):
    # Use textbbox to get the bounding box of the text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return text_width, text_height


def get_max_font_size(text, font_path, num_lines=1, max_margin=64, image_size=512, initial_font_size=10):
    background = Image.new('RGB', (image_size, image_size), 'black')
    background = ImageDraw.Draw(background)

    # Start with a reasonably large font size for initial measurement
    # print('font_path', font_path, 'initial_font_size', initial_font_size)
    initial_font = ImageFont.truetype(font_path, initial_font_size)
    initial_text_width, initial_text_height = get_text_dimensions(background, text, initial_font)
    # initial_text_height *= num_lines
    if initial_text_width == 0 or initial_text_height == 0:
        return -1

    # Calculate the scaling factor for both width and height
    scale_factor_width = (image_size - 2 * max_margin) / initial_text_width
    scale_factor_height = ((image_size - 2 * max_margin) / num_lines) / initial_text_height

    # Use the smaller scaling factor to ensure the text fits within both dimensions
    scale_factor = min(scale_factor_width, scale_factor_height)

    # Calculate the maximum font size
    max_font_size = int(initial_font_size * scale_factor)
    new_text_width, new_text_height = get_text_dimensions(background, text, ImageFont.truetype(font_path, max_font_size))

    # Ensure the font size is not less than 1
    return max(1, max_font_size - 3), new_text_width, new_text_height


def get_random_text(len):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(len))


def generate_texts(content_zh, content_en, min_length=3, max_length=6):
    # Select a random text
    if random.random() < 0.8:
        # Randomly choose a length between min_length and max_length
        if random.random() < 0.5:
            start_index = random.randint(0, len(content_zh) - min_length)
            chosen_length = random.randint(min_length, max_length)
            chosen_length = min(chosen_length, len(content_zh) - start_index)

            text = content_zh[start_index:start_index + chosen_length]
            flag_chinese = True
        else:
            start_index = random.randint(0, len(content_en) - min_length)
            chosen_length = random.randint(min_length, max_length)
            chosen_length = min(chosen_length, len(content_en) - start_index)

            text = content_en[start_index:start_index + chosen_length]
            flag_chinese = False
    else:
        text = get_random_text(random.randint(3, 10))
        flag_chinese = False

    return text, flag_chinese


def load_texts():
    """
    Load texts from a text file.
    """
    # Read the content of the file
    annotation_file = 'nejm_test_zh.txt'
    with open(annotation_file, 'r', encoding='utf-8') as file:
        content_zh = file.read()

    annotation_file = 'nejm_test_en.txt'
    with open(annotation_file, 'r', encoding='utf-8') as file:
        content_en = file.read()

    content_zh = content_zh.replace(' ', '').replace('\n', '')
    content_en = content_en.replace(' ', '').replace('\n', '')
    return content_zh, content_en

    # with open(annotation_file, 'r') as file:
    #     annotations = json.load(file)
    #
    # all_texts = {}
    # for id, annot in annotations.items():
    #     if annot['illegibility'] == 'True' or annot['illegibility'] == 'true' \
    #         or annot['illegibility'] == True or annot['transcription'] == '###' or len(annot['transcription']) < 2:
    #         continue
    #     all_texts[annot['transcription']] = annot['language']
    # return all_texts


def gaussian(x, mu, sigma):
    return math.exp(-math.pow(x - mu, 2.) / (2 * math.pow(sigma, 2.)))


def render_clean_text_image(image_size, text, font_path, font_size, no_font=False):
    """
    Render clean text centered on an image.
    """
    font = ImageFont.truetype(font_path, font_size)
    clean_img = Image.new('RGB', image_size, 'black')
    draw = ImageDraw.Draw(clean_img)

    _, _, text_width, text_height = draw.textbbox((0, 0), text, font=font)
    text_position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

    draw.text(text_position, text, font=font, fill=(255, 255, 255))

    # Create an image for the rectangle
    rectangle_img = Image.new('RGB', image_size, 'black')
    rect_draw = ImageDraw.Draw(rectangle_img)
    circle_diameter = 10  # You can adjust the size of the circles here
    sigma = circle_diameter / 3  # Standard deviation for Gaussian blur

    top_left = text_position
    top_right = (text_position[0] + text_width, text_position[1])
    bottom_left = (text_position[0], text_position[1] + text_height)
    bottom_right = (text_position[0] + text_width, text_position[1] + text_height)
    corners = [top_left, top_right, bottom_left, bottom_right]

    # Draw Gaussian dots at each corner
    for corner in corners:
        for dx in range(-circle_diameter, circle_diameter):
            for dy in range(-circle_diameter, circle_diameter):
                dist = math.sqrt(dx ** 2 + dy ** 2)
                if dist <= circle_diameter:
                    intensity = int(255 * gaussian(dist, 0, sigma))
                    color = (intensity, intensity, intensity)  # Grayscale intensity
                    x = corner[0] + dx
                    y = corner[1] + dy
                    if 0 <= x < image_size[0] and 0 <= y < image_size[1]:  # Check bounds
                        rect_draw.point((x, y), fill=color)
                        draw.point((x, y), fill=(150, 200, 10))
    # rect_draw.rectangle([top_left, bottom_right], outline="white", width=2)

    # Create an image for the horizontal line
    line_img = Image.new('RGB', image_size, 'black')
    line_draw = ImageDraw.Draw(line_img)
    center_y = text_position[1] + text_height // 2
    line_draw.line([(0, center_y), (image_size[0], center_y)], width=2)

    return clean_img, rectangle_img, line_img, corners


def render_clean_text_image_multilines(image_size, text, font_path_all_lines, font_size_all_lines, annot_each_line=True):
    """
    Render clean text centered on an image, supporting multiple lines.
    """
    clean_img = Image.new('RGB', image_size, 'black')
    draw = ImageDraw.Draw(clean_img)
    rectangle_img = Image.new('RGB', image_size, 'black')
    rect_draw = ImageDraw.Draw(rectangle_img)
    line_img = Image.new('RGB', image_size, 'black')
    line_draw = ImageDraw.Draw(line_img)
    circle_diameter = 10  # You can adjust the size of the circles here
    sigma = circle_diameter / 3  # Standard deviation for Gaussian blur

    # Split the text into lines
    lines = text.split('\n')
    total_text_height = 0
    max_text_width = 0
    line_widths = []
    line_heights = []

    # Calculate total height of text and individual line heights
    for i, line in enumerate(lines):
        if len(font_path_all_lines) == 1:
            font = ImageFont.truetype(font_path_all_lines[0], font_size_all_lines[i])
        else:
            font = ImageFont.truetype(font_path_all_lines[i], font_size_all_lines[i])

        _, _, text_width, text_height = draw.textbbox((0, 0), line, font=font)
        total_text_height += text_height
        line_widths.append(text_width)
        line_heights.append(text_height)
        max_text_width = max(max_text_width, text_width)

    # Calculate starting Y position
    y = (image_size[1] - total_text_height) // 2

    if annot_each_line:
        # Draw each line of text and associated graphics
        for i, line in enumerate(lines):
            if len(font_path_all_lines) == 1:
                font = ImageFont.truetype(font_path_all_lines[0], font_size_all_lines[i])
            else:
                font = ImageFont.truetype(font_path_all_lines[i], font_size_all_lines[i])

            text_position = ((image_size[0] - max_text_width) // 2, y)
            draw.text(text_position, line, font=font, fill=(255, 255, 255))

            # Update the Y position for the next line
            y += line_heights[i]

            # Gaussian points for each line
            top_left = text_position
            top_right = (text_position[0] + line_widths[i], text_position[1])
            bottom_left = (text_position[0], text_position[1] + line_heights[i])
            bottom_right = (text_position[0] + line_widths[i], text_position[1] + line_heights[i])
            corners = [top_left, top_right, bottom_left, bottom_right]

            for corner in corners:
                for dx in range(-circle_diameter, circle_diameter):
                    for dy in range(-circle_diameter, circle_diameter):
                        dist = math.sqrt(dx ** 2 + dy ** 2)
                        if dist <= circle_diameter:
                            intensity = int(255 * gaussian(dist, 0, sigma))
                            x = corner[0] + dx
                            y = corner[1] + dy
                            if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                                # if overlap, add pixel values, do not overwrite
                                current_color = rectangle_img.getpixel((x, y))
                                new_intensity = min(255, current_color[0] + intensity)
                                color = (new_intensity, new_intensity, new_intensity)
                                rect_draw.point((x, y), fill=color)
                                draw.point((x, y), fill=(150, 200, 10))

            # Midline for each line
            center_y = (top_left[1] + bottom_right[1]) // 2
            line_draw.line([(0, center_y), (image_size[0], center_y)], width=2)

        # Calculate corners for the entire text block
        top_left = ((image_size[0] - max_text_width) // 2, (image_size[1] - total_text_height) // 2)
        top_right = ((image_size[0] - max_text_width) // 2, (image_size[1] + total_text_height) // 2)
        bottom_left = ((image_size[0] + max_text_width) // 2, (image_size[1] - total_text_height) // 2)
        bottom_right = ((image_size[0] + max_text_width) // 2, (image_size[1] + total_text_height) // 2)
        corners = [top_left, top_right, bottom_left, bottom_right]

    else:
        # Draw each line of text
        for i, line in enumerate(lines):
            if len(font_path_all_lines) == 1:
                font = ImageFont.truetype(font_path_all_lines[0], font_size_all_lines[i])
            else:
                font = ImageFont.truetype(font_path_all_lines[i], font_size_all_lines[i])

            text_position = ((image_size[0] - max_text_width) // 2, y)
            draw.text(text_position, line, font=font, fill=(255, 255, 255))

            # Update the Y position for the next line
            y += line_heights[i]

        # Create an image for the rectangle
        top_left = ((image_size[0] - max_text_width) // 2, (image_size[1] - total_text_height) // 2)
        top_right = ((image_size[0] - max_text_width) // 2, (image_size[1] + total_text_height) // 2)
        bottom_left = ((image_size[0] + max_text_width) // 2, (image_size[1] - total_text_height) // 2)
        bottom_right = ((image_size[0] + max_text_width) // 2, (image_size[1] + total_text_height) // 2)
        corners = [top_left, top_right, bottom_left, bottom_right]

        # Draw Gaussian dots at each corner
        for corner in corners:
            for dx in range(-circle_diameter, circle_diameter):
                for dy in range(-circle_diameter, circle_diameter):
                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    if dist <= circle_diameter:
                        intensity = int(255 * gaussian(dist, 0, sigma))
                        color = (intensity, intensity, intensity)  # Grayscale intensity
                        x = corner[0] + dx
                        y = corner[1] + dy
                        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:  # Check bounds
                            rect_draw.point((x, y), fill=color)
                            # draw.point((x, y), fill=(150, 200, 10))
        # rect_draw.rectangle([top_left, bottom_right], outline="white", width=2)

        # Create an image for the horizontal line
        center_y = (top_left[1] + bottom_right[1]) // 2
        line_draw.line([(0, center_y), (image_size[0], center_y)], width=2)

    return clean_img, rectangle_img, line_img, corners


def generate_random_color():
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))


def add_noise_and_color_to_image(image, noise_intensity=10, jitter_intensity=0.1):
    """
    Add random noise and perform color jittering on the given PIL image.

    :param image: Input PIL image.
    :param noise_intensity: Intensity of the noise to add.
    :param jitter_intensity: Intensity of the color jittering.
    """
    # Convert to numpy array for noise addition
    np_image = np.array(image)

    # Adding noise
    for i in range(3):  # Assuming RGB image
        noise = np.random.randint(-noise_intensity, noise_intensity, (np_image.shape[0], np_image.shape[1]), dtype='int16')
        np_image[..., i] = np.clip(np_image[..., i] + noise, 0, 255)

    noisy_image = Image.fromarray(np_image.astype('uint8'), 'RGB')

    # Color jittering
    brightness_factor = random.uniform(1 - jitter_intensity, 1 + jitter_intensity)
    contrast_factor = random.uniform(1 - jitter_intensity, 1 + jitter_intensity)
    saturation_factor = random.uniform(1 - jitter_intensity, 1 + jitter_intensity)
    hue_factor = random.uniform(-jitter_intensity, jitter_intensity)

    noisy_image = ImageEnhance.Brightness(noisy_image).enhance(brightness_factor)
    noisy_image = ImageEnhance.Contrast(noisy_image).enhance(contrast_factor)
    noisy_image = ImageEnhance.Color(noisy_image).enhance(saturation_factor)

    # Hue adjustment requires conversion to HSV
    hsv_image = noisy_image.convert('HSV')
    np_hsv_image = np.array(hsv_image)
    np_hsv_image[..., 0] = np_hsv_image[..., 0] + int(hue_factor * 255)
    np_hsv_image[..., 0] = np.clip(np_hsv_image[..., 0], 0, 255)
    noisy_image = Image.fromarray(np_hsv_image, 'HSV').convert('RGB')

    return noisy_image


def apply_random_curvatures(image, rectangle_img, line_img):
    """
    Apply versatile curvature to the given text image using a sine wave.

    :param image: PIL image with text.
    :return: PIL image with curved text.
    """
    # Convert PIL image to OpenCV format
    image = np.array(image)
    rectangle_img = np.array(rectangle_img)
    line_img = np.array(line_img)

    # Get image dimensions
    height, width = image.shape[:2]

    # Create a new image that can accommodate the curvature
    curved_image = np.zeros_like(image)
    curved_rectangle_img = np.zeros_like(rectangle_img)
    curved_line_img = np.zeros_like(line_img)

    # Generate random curvature parameters
    """
    param amplitude: Amplitude of the sine wave, controlling the depth of the curve.
    param frequency: Frequency of the sine wave, controlling how many curves there will be.
    param phase_shift: Phase shift of the sine wave, shifting the curve horizontally.
    """
    amplitude = random.randint(0, 30)
    frequency = random.uniform(0.5, 2)
    phase_shift = random.uniform(0, 2 * np.pi)

    # Apply curvature by repositioning each pixel
    for y in range(height):
        for x in range(width):
            new_x = x
            new_y = int(y + amplitude * np.sin(2 * np.pi * frequency * (x / width) + phase_shift))

            # Ensure new_y is within the image bounds
            if 0 <= new_y < height:
                curved_image[new_y, new_x] = image[y, x]
                curved_rectangle_img[new_y, new_x] = rectangle_img[y, x]
                curved_line_img[new_y, new_x] = line_img[y, x]

    # Convert back to PIL format
    curved_image = Image.fromarray(curved_image)
    curved_rectangle_img = Image.fromarray(curved_rectangle_img)
    curved_line_img = Image.fromarray(curved_line_img)

    return curved_image, curved_rectangle_img, curved_line_img


def apply_random_transformations(image, rectangle_img, line_img, text_corners, max_rotation=60, perspective_variation=0.0005, margin=96):
    # Load image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rectangle_img = cv2.cvtColor(np.array(rectangle_img), cv2.COLOR_RGB2BGR)
    line_img = cv2.cvtColor(np.array(line_img), cv2.COLOR_RGB2BGR)
    image_height, image_width = image.shape[:2]

    # Define the center of the image
    cx, cy = image_width / 2.0, image_height / 2.0

    # Random rotation
    rotation_angle_deg = random.uniform(-max_rotation, max_rotation)
    rotation_angle_rad = np.deg2rad(rotation_angle_deg)
    cos_angle = np.cos(rotation_angle_rad)
    sin_angle = np.sin(rotation_angle_rad)

    rotation_matrix = np.array([
        [cos_angle, -sin_angle, cx * (1 - cos_angle) + cy * sin_angle],
        [sin_angle, cos_angle, cy * (1 - cos_angle) - cx * sin_angle],
        [0, 0, 1]
    ])

    # Random perspective transformation
    px = random.uniform(-perspective_variation, perspective_variation)
    py = random.uniform(-perspective_variation, perspective_variation)

    perspective_matrix = np.array([
        [1, px, 0],
        [py, 1 + px * py, 0],
        [px * py, px + py, 1]
    ])

    # Combine transformations
    transformation_matrix = np.dot(perspective_matrix, rotation_matrix)

    # Transform the corners of the original image to find new bounds
    original_corners = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]], dtype=np.float32)
    original_corners = np.array([original_corners])
    transformed_image_corners = cv2.perspectiveTransform(original_corners, transformation_matrix)

    # Calculate required shift to bring all coordinates into positive space
    x_min_img = np.min(transformed_image_corners[:, :, 0])
    y_min_img = np.min(transformed_image_corners[:, :, 1])
    x_max_img = np.max(transformed_image_corners[:, :, 0])
    y_max_img = np.max(transformed_image_corners[:, :, 1])

    dx = -min(0, x_min_img)
    dy = -min(0, y_min_img)
    new_width_img = int(x_max_img + dx)
    new_height_img = int(y_max_img + dy)

    # Update transformation matrix to include the shift
    translation_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    transformation_matrix = np.dot(translation_matrix, transformation_matrix)

    # Apply the adjusted transformation
    transformed_image = cv2.warpPerspective(image, transformation_matrix, (new_width_img, new_height_img))
    transformed_rectangle_img = cv2.warpPerspective(rectangle_img, transformation_matrix, (new_width_img, new_height_img))
    transformed_line_img = cv2.warpPerspective(line_img, transformation_matrix, (new_width_img, new_height_img))

    # Transform the text corners with the adjusted matrix
    text_corners = np.array([text_corners], dtype=np.float32)
    transformed_text_corners = cv2.perspectiveTransform(text_corners, transformation_matrix)

    # Calculate new bounds with margins, ensuring they are within the image limits
    x_min_text = max(0, np.min(transformed_text_corners[:, :, 0]) - margin)
    y_min_text = max(0, np.min(transformed_text_corners[:, :, 1]) - margin)
    x_max_text = min(new_width_img, np.max(transformed_text_corners[:, :, 0]) + margin)
    y_max_text = min(new_height_img, np.max(transformed_text_corners[:, :, 1]) + margin)

    # Crop the text region
    transformed_image = transformed_image[int(y_min_text):int(y_max_text), int(x_min_text):int(x_max_text)]
    transformed_rectangle_img = transformed_rectangle_img[int(y_min_text):int(y_max_text), int(x_min_text):int(x_max_text)]
    transformed_line_img = transformed_line_img[int(y_min_text):int(y_max_text), int(x_min_text):int(x_max_text)]

    # resize back to original image size
    transformed_image = cv2.resize(transformed_image, (image_width, image_height))
    transformed_rectangle_img = cv2.resize(transformed_rectangle_img, (image_width, image_height))
    transformed_line_img = cv2.resize(transformed_line_img, (image_width, image_height))

    return transformed_image, transformed_rectangle_img, transformed_line_img


def overlay_on_random_background(transformed_text_img, image_folder):
    """
    Overlay the transformed text image on a random background image.
    """
    # Load a random background image
    background_image_path = random.choice(os.listdir(image_folder))
    background_img = Image.open(os.path.join(image_folder, background_image_path)).convert("RGB").resize((512, 512))

    # Calculate medium grey and reduce contrast of the background image
    background_img = np.array(background_img)
    medium_grey = np.full(background_img.shape, 128)
    background_img = background_img * 0.7 + medium_grey * 0.3
    background_img = np.clip(background_img, 0, 255).astype(np.uint8)
    background_img = Image.fromarray(background_img)

    # Create a mask for blending
    mask = transformed_text_img.convert("L")

    # Generate a random color
    new_color = generate_random_color()
    transformed_text_img = transformed_text_img.convert("RGB")
    pixels = transformed_text_img.load()

    for i in range(transformed_text_img.width):
        for j in range(transformed_text_img.height):
            r, g, b = pixels[i, j]
            if (r, g, b) != (0, 0, 0):  # If the pixel is not black
                pixels[i, j] = new_color

    # Overlay the text image on the background
    background_img.paste(transformed_text_img, (0, 0), mask)

    return background_img


def generate_text_image_pairs(num_pairs, image_size, fonts_dir, output_dir, background_dir, offset=0, save_recovered=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load all candidate texts from the annotation file
    content_zh, content_en = load_texts()

    previous_num_lines, previous_text_all_lines = None, None

    for i in tqdm(range(3 * num_pairs)):
        text_subset_idx = i // 3
        font_variation = i % 3
        if i % 3 == 0:  # for each text, generate three different fonts for contrastive learning
            text_all_lines = ""
            font_path_all_lines = []
            font_size_all_lines = []
            flag_chinese_all_lines = []
            texts_all_lines = []

            # if random.random() < 0.3:
            num_lines = random.randint(2, 4)
            # else:
            #     num_lines = 1

            for l in range(num_lines):
                text, flag_chinese = generate_texts(content_zh, content_en)
                flag_chinese_all_lines.append(flag_chinese)
                texts_all_lines.append(text)

                if len(text) > 10:
                    text = text[:10]
                if len(text_all_lines) == 0:
                    text_all_lines += text
                else:
                    text_all_lines += '\n' + text

                # Select a random font
                font_path = load_fonts(fonts_dir, chinese=flag_chinese)
                # print('text', text, 'font_path', font_path)
                font_size, text_width, text_height = get_max_font_size(text, font_path, num_lines)
                if font_size == -1:
                    continue
                font_path_all_lines.append(font_path)
                font_size_all_lines.append(font_size)

            previous_num_lines = num_lines
            previous_text_all_lines = text_all_lines

        else:
            num_lines = previous_num_lines
            text_all_lines = previous_text_all_lines

            font_path_all_lines = []
            font_size_all_lines = []
            for l in range(num_lines):
                # Select a random font
                font_path = load_fonts(fonts_dir, chinese=flag_chinese_all_lines[l])
                # print('text', text, 'font_path', font_path)
                font_size, text_width, text_height = get_max_font_size(texts_all_lines[l], font_path, num_lines)
                if font_size == -1:
                    continue
                font_path_all_lines.append(font_path)
                font_size_all_lines.append(font_size)

        # Render a clean text images
        if num_lines == 1:
            clean_img, clean_rectangle_img, clean_line_img, corners = render_clean_text_image(image_size, text_all_lines, font_path_all_lines[0], font_size_all_lines[0])
        else:
            try:
                clean_img, clean_rectangle_img, clean_line_img, corners = render_clean_text_image_multilines(image_size, text_all_lines, font_path_all_lines, font_size_all_lines)
            except:
                num_lines = 1
                clean_img, clean_rectangle_img, clean_line_img, corners = render_clean_text_image(image_size, text_all_lines, font_path_all_lines[0], font_size_all_lines[0])

        # Render a curved, perspective transformed, noisy, and color jittered text
        # It is okay if transformed texts miss parts of their characters, because the meaning of texts do not matter
        # curved_text_img, curved_rectangle_img, curved_line_img = apply_random_transformations(clean_img, clean_rectangle_img, clean_line_img, corners)
        # curved_text_img, curved_rectangle_img, curved_line_img = apply_random_curvatures(curved_text_img, curved_rectangle_img, curved_line_img)
        curved_text_img, curved_rectangle_img, curved_line_img = clean_img, clean_rectangle_img, clean_line_img


        if save_recovered:
            corners = detect_gaussian_corners(np.asarray(curved_rectangle_img), num_lines, os.path.join(output_dir + "/recovered", f"{text_subset_idx + offset}_{font_variation}"))
            if corners is not None:
                # continue
                corners = order_corners(np.asarray(curved_line_img), corners)
                target_corners = create_rectangle(image_size, text_width, text_height * num_lines)
                corrected_image = perform_perspective_transform(np.asarray(curved_text_img), image_size, corners, target_corners)
                corrected_line_img = perform_perspective_transform(np.asarray(curved_line_img), image_size, corners, target_corners)

                # Save the recovered images
                recovered_text_img_path = os.path.join(output_dir + "/recovered", f"{text_subset_idx + offset}_{font_variation}_after_persp.png")
                Image.fromarray(corrected_image).save(recovered_text_img_path)
                recovered_line_img_path = os.path.join(output_dir + "/recovered", f"{text_subset_idx + offset}_{font_variation}_line_after_persp.png")
                Image.fromarray(corrected_line_img).save(recovered_line_img_path)

                midline_start, midline_end = find_min_max_x_coordinates(corrected_line_img)
                straight_line_img = Image.new("RGB", image_size, (0, 0, 0))
                draw = ImageDraw.Draw(straight_line_img)
                draw.line([midline_start, midline_end], fill=(255, 255, 255), width=2)
                straight_line_img_path = os.path.join(output_dir + "/recovered", f"{text_subset_idx + offset}_{font_variation}_straight_line.png")
                straight_line_img.save(straight_line_img_path)

                corrected_image = correct_curvature(corrected_image, corrected_line_img, image_size, midline_start, midline_end)

                recovered_text_img_path = os.path.join(output_dir + "/recovered", f"{text_subset_idx + offset}_{font_variation}.png")
                Image.fromarray(corrected_image).save(recovered_text_img_path)
            else:
                continue
        else:
            corners = detect_gaussian_corners(np.asarray(curved_rectangle_img), num_lines)
            if corners is None:
                continue    # ensure each image in our dataset has four valid corners being detected


        curved_text_img_target_path = os.path.join(output_dir + "/target_curved", f"{text_subset_idx + offset}_{font_variation}.png")
        curved_text_img.save(curved_text_img_target_path)

        curved_text_img = overlay_on_random_background(curved_text_img, background_dir)
        if random.random() < 0.7:
            curved_text_img = add_noise_and_color_to_image(curved_text_img)

        # Save the images
        clean_img_path = os.path.join(output_dir + "/target", f"{text_subset_idx + offset}_{font_variation}.png")
        curved_text_img_path = os.path.join(output_dir + "/source", f"{text_subset_idx + offset}_{font_variation}.png")
        clean_img.save(clean_img_path)
        curved_text_img.save(curved_text_img_path)

        rectangle_img_path = os.path.join(output_dir + "/corners", f"{text_subset_idx + offset}_{font_variation}.png")
        curved_rectangle_img.save(rectangle_img_path)
        line_img_path = os.path.join(output_dir + "/midline", f"{text_subset_idx + offset}_{font_variation}.png")
        curved_line_img.save(line_img_path)

        texts_path = os.path.join(output_dir + "/texts", f"{text_subset_idx + offset}_{font_variation}.txt")
        with open(texts_path, 'w') as file:
            file.write(text_all_lines)
        fonts_path = os.path.join(output_dir + "/fonts", f"{text_subset_idx + offset}_{font_variation}.txt")
        with open(fonts_path, 'w') as file:
            for font_path in font_path_all_lines:
                file.write(font_path + '\n')


if __name__ == "__main__":
    # Parameters for generating the synthetic dataset
    offset = 0
    num_pairs = 10-offset
    image_size = (512, 512)
    fonts_dir = "fonts"
    output_dir = "toy_examples"
    background_dir = "/tmp/datasets/coco/train2017"

    os.makedirs(output_dir + "/target", exist_ok=True)
    os.makedirs(output_dir + "/target_curved", exist_ok=True)
    os.makedirs(output_dir + "/corners", exist_ok=True)
    os.makedirs(output_dir + "/midline", exist_ok=True)
    # os.makedirs(output_dir + "/no_font", exist_ok=True)
    os.makedirs(output_dir + "/recovered", exist_ok=True)
    os.makedirs(output_dir + "/source", exist_ok=True)
    # os.makedirs(output_dir + "/coeffs", exist_ok=True)
    os.makedirs(output_dir + "/texts", exist_ok=True)
    os.makedirs(output_dir + "/fonts", exist_ok=True)

    # Generate the text image pairs
    generate_text_image_pairs(num_pairs, image_size, fonts_dir, output_dir, background_dir, offset, save_recovered=False)
