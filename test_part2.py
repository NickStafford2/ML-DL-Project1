# reads from the cmdline and file and load saved model and run predict

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("./best_model_part2.keras")

image_name = input(
    "enter the name of your image placed in root directory(with extension):"
)

image_path = f"ML-DL-Project1/pyrightconfig.json"
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
