import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("../models/digit_cnn.h5")

img = cv2.imread("sample_digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img.reshape(1, 28, 28, 1) / 255.0

pred = model.predict(img)
print("Predicted digit:", np.argmax(pred))
