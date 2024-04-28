import cv2
import numpy as np


def order_points(pts):
    # Initial sorting based on the x-coordinate
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # Get the left-most and right-most points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # Sort left-most coordinates according to their y-coordinates
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # Sort the right-most coordinates according to their y-coordinates
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    # Return the coordinates in top-left, top-right, bottom-right, bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


# Load image
image = cv2.imread('toy_examples/target_curved/0.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Denoise and threshold
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Detect edges and find contours
edges = cv2.Canny(thresh, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

text_contour = contours[0]

# Approximate the contour to a quadrilateral
epsilon = 0.1 * cv2.arcLength(text_contour, True)
approx = cv2.approxPolyDP(text_contour, epsilon, True)

# Draw the contour (for visualization)
cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)

# Save or display the image
cv2.imwrite('image_with_quadrilateral.png', image)

# # Calculate the bounding box for all contours and draw it
# x_min = min([cv2.boundingRect(contour)[0] for contour in contours])
# y_min = min([cv2.boundingRect(contour)[1] for contour in contours])
# x_max = max([cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] for contour in contours])
# y_max = max([cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3] for contour in contours])
#
# # Draw bounding rectangle around the whole text
# cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#
# # Save or display the image
# cv2.imwrite('image_with_bounding_box.png', image)
#
# # Perspective correction (assuming approx is a quadrilateral)
# src_pts = np.array([
#     [x_min, y_min],
#     [x_max, y_min],
#     [x_max, y_max],
#     [x_min, y_max]
# ], dtype='float32')
# dst_pts = np.array([[0, 0], [512, 0], [512, 512], [0, 512]], dtype='float32')  # Adjust size as needed
# print('Source points:', src_pts, '\nDestination points:', dst_pts)
# matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
# warped = cv2.warpPerspective(gray, matrix, (512, 512))
# cv2.imwrite('warped.png', warped)
#
# # Inverting colors and changing to white text on black background
# _, final_binary = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY_INV)
#
# # Save the result to a file
# cv2.imwrite('rectified_text.png', final_binary)
