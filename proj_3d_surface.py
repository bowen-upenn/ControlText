import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# !pip install timm

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Load an image
bg_img = cv2.imread('hat.png')
bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

input_batch = transform(bg_img).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=bg_img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth = prediction.cpu().numpy()
depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
depth = 2 - depth

plt.imshow(depth)
plt.colorbar(label='Depth')

txt_img = cv2.imread('texts.png')[:, :, 0]
plt.imshow(txt_img)

# Create 3D points
h, w = depth.shape
x, y = np.meshgrid(np.arange(w), np.arange(h))
z = depth

# Stack into a (N, 3) array of 3D points
points_3d = np.stack((x, y, z), axis=-1).reshape(-1, 3)

# Define camera parameters
focal_length = 1  # Adjust based on the desired perspective
camera_matrix = np.array([[focal_length, 0, 0.0],
                          [0, focal_length, 0.0],
                          [0, 0, 1]])

# Project 3D points to 2D
points_2d, _ = cv2.projectPoints(points_3d, (0, 0, 0), (0, 0, 0), camera_matrix, None)

# Reshape to image dimensions
points_2d = points_2d.reshape(h, w, 2)

# Create a distorted image
distorted_image = np.zeros_like(txt_img)

# # Map original image pixels to new positions
# for i in range(h):
#     for j in range(w):
#         x_new, y_new = points_2d[i, j]
#         x_new = int(x_new)
#         y_new = int(y_new)
#         if 0 <= x_new < w and 0 <= y_new < h:
#             distorted_image[y_new, x_new] = txt_img[i, j]

# Create an inverse map
inverse_map = np.full((h, w, 2), -1.0)

# Populate the inverse map
for i in range(h):
    for j in range(w):
        x_new, y_new = points_2d[i, j]
        x_new = int(x_new)
        y_new = int(y_new)
        if 0 <= x_new < w and 0 <= y_new < h:
            inverse_map[y_new, x_new] = [i, j]

# Fill the distorted image using the inverse map
for i in range(h):
    for j in range(w):
        src_y, src_x = inverse_map[i, j]
        if src_y >= 0 and src_x >= 0:
            distorted_image[i, j] = txt_img[int(src_y), int(src_x)]


print('distorted_image', distorted_image.shape, np.min(distorted_image), np.max(distorted_image))

# Display the result
plt.imshow(distorted_image)


