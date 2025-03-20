import cv2
import numpy as np

from project1.split_data import Datasets


def image_path_to_tensor(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image.shape
    return image
