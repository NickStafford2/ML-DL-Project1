import cv2
import numpy as np


def image_path_to_tensor(image_path: str) -> np.ndarray:
    print(f"converting: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    image_tensor = np.asarray(image)

    image_tensor = image_tensor / 255
    image_tensor = image_tensor.reshape(224, 224, 1)  # reshaping for input to keras

    return image_tensor


# def get_shape(image_path: str):
#     # adding a 3rd dimension for channel(grayscale) since keras expects batches of images
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image_batch = image.reshape(1, image.shape[0], image.shape[1], 1)
#     return image_batch.shape[1:]
