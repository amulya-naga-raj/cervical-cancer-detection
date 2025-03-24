import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Median Blur
    blur = cv2.medianBlur(gray, 5)

    # Sharpening using kernel
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(blur, -1, kernel)

    # Thresholding
    ret, thresh = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)

    return thresh
