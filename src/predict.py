import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("model/cnn_model.h5")

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (180, 180))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)

    stage = np.argmax(prediction)

    return f"Predicted Cancer Stage: Stage {stage + 1}"
