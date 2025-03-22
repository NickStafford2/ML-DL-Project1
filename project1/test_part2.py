# reads from the cmdline and file and load saved model and run predict

import PIL
import cv2
import numpy as np

# from somewhere import saved_model_pt1
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# model = load_model("best_model.zip")
model = load_model("../best_model.keras")

image_name = input(
    "enter the name of your image placed in root directory(with extension):"
)
image_path = f"/workspaces/ML-DL-Project1/{image_name}"
# image = Image.open(image_path)
# image = image.resize()
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
image = image.reshape(1, 224, 224, 3)

image_tensor = np.asarray(image)
print(image_tensor.shape)
probability_values = model.predict(image_tensor)
predicted_class = np.argmax(probability_values)
class_names = {
    0: "Class 1 Hands with Touch",
    1: "Class 2 No hands",
    2: "Class 3 Hands without Touch",
}
print("predicted class: ", class_names[int(predicted_class)])
