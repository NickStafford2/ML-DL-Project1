import cv2
import numpy as np

from project1.split_data import Datasets


def image_path_to_tensor(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    image_tensor = np.asarray(image)
    image_tensor = image_tensor / 255
    image_tensor = image_tensor.reshape(224, 224, 1)  # reshaping for input to keras

    return image_tensor
