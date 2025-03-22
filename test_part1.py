# reads from the cmdline and file and load saved model and run predict

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("./best_model_part1.keras")

image_name = input(
    "enter the name of your image placed in root directory(with extension):"
)

# replace this with your local path
image_path = f"ML-DL-Project1/pyrightconfig.json"
image = cv2.imread(image_path)
image = cv2.resize(image, (48, 48))
image = image.reshape(1, 48, 48, 1)

image_tensor = np.asarray(image)
print(image_tensor.shape)
probability_values = model.predict(image_tensor)
predicted_class = np.argmax(probability_values)
class_names = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}
print("predicted class: ", class_names[int(predicted_class)])
