# augmentation_module.py

import cv2
import numpy as np

import cv2
import numpy as np

def horizontal_flip(image, bboxes):
    """
    Horizontally flips the image .

    Parameters:
    - image: numpy array of the image.
    - bboxes: list of bounding boxes in [x_min, y_min, x_max, y_max] format.

    Returns:
    - flipped_image: The horizontally flipped image
    - flipped_bboxes: Adjusted bounding boxes 
    """
    flipped_image = cv2.flip(image, 1)
    image_width = image.shape[1]

    flipped_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        new_x_min = image_width - x_max
        new_x_max = image_width - x_min
        flipped_bboxes.append([new_x_min, y_min, new_x_max, y_max])

    return flipped_image, flipped_bboxes

def gaussian_blur(image, bboxes, kernel_size=(5, 5)):
    """
    Applies Gaussian blur

    Parameters:
    - image: numpy array
    - bboxes: list of bounding boxes
    - kernel_size: Size of the Gaussian kernel

    Returns:
    - blurred_image: The image after Gaussian blur
    - bboxes: Unchanged bounding boxes
    """
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image, bboxes


def resize_image(image, bboxes, new_size):
    """
    Resizes the image, adjusts bounding boxes.

    Parameters:
    - image: numpy array 
    - bboxes: list of bounding boxes
    - new_size: New size as (width, height)

    Returns:
    - resized_image:  resized image.
    - resized_bboxes: Adjusted bounding boxes.
    """
    original_size = image.shape[1], image.shape[0]  # (width, height)
    resized_image = cv2.resize(image, new_size)
    x_scale = new_size[0] / original_size[0]
    y_scale = new_size[1] / original_size[1]

    resized_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        new_x_min = x_min * x_scale
        new_x_max = x_max * x_scale
        new_y_min = y_min * y_scale
        new_y_max = y_max * y_scale
        resized_bboxes.append([new_x_min, new_y_min, new_x_max, new_y_max])

    return resized_image, resized_bboxes

def rotate_image(image, bboxes, angle):
    """
    Rotates the image, adjusts bounding boxes.

    Parameters:
    - image: numpy array of the image
    - bboxes: list of bounding boxes
    - angle: Angle to rotate the image.

    Returns:
    - rotated_image:  rotated image
    - rotated_bboxes: Adjusted bounding boxes .
    """
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])

    # Compute new dimensions
    new_width = int((height * sin_theta) + (width * cos_theta))
    new_height = int((height * cos_theta) + (width * sin_theta))

    # Adjust rotation matrix
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    # Adjust bounding boxes
    rotated_bboxes = []
    for bbox in bboxes:
        # Get the coordinates of the corners
        corners = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]]
        ])
        ones = np.ones(shape=(len(corners), 1))
        corners_ones = np.hstack([corners, ones])

        # Apply rotation
        transformed_corners = rotation_matrix.dot(corners_ones.T).T
        x_coords = transformed_corners[:, 0]
        y_coords = transformed_corners[:, 1]

        # Get new bounding box
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        rotated_bboxes.append([x_min, y_min, x_max, y_max])

    return rotated_image, rotated_bboxes

def adjust_brightness(image, bboxes, factor=1.0):
    """
    Adjusts image brightness

    Parameters:
    - image: numpy array 
    - bboxes: list of bounding boxes (unchanged)
    - factor: Brightness factor (>1 increases brightness, <1 decreases brightness)

    Returns:
    - bright_image: The image after changing brightness.
    - bboxes: Unchanged bounding boxes
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] *= factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255  # Cap values
    hsv = np.array(hsv, dtype=np.uint8)
    bright_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bright_image, bboxes


def add_noise(image, bboxes, mean=0, var=0.01):
    """
    Adds Gaussian noise

    Parameters:
    - image: numpy array
    - bboxes: list of bounding boxes
    - mean: Mean Gaussian noise.
    - var: Variance of Gaussian noise.

    Returns:
    - noisy_image: The image with noise.
    - bboxes: Unchanged bounding boxes.
    """
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv2.add(image, gauss)
    return noisy_image, bboxes