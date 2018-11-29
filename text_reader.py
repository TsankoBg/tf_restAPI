import numpy as np
import cv2
import config
import pytesseract
from pytesseract import image_to_string


class ImageTextReader:

    def readText(self, image):
        # Convert to gray
        temp = np.asarray(image)
        img = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        # Apply dilation and erosion to remove some noise
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        #  Apply threshold to get image with only black and white
        #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        # Recognize text with tesseract for python
        result = pytesseract.image_to_string(img)
        return str(result)
