import numpy as np
import cv2
import config
import pytesseract
from pytesseract import image_to_string


class ImageTextReader:

#300 dpi minimum to be detected 
#text around 12pt
#fix text lines
#image without dark parts
    def readText(self, image):
        # Convert to gray
        temp = np.asarray(image)
        img = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        # Apply dilation and erosion to remove some noise
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        cv2.imwrite( "thres.png", img)
        #  Apply threshold to get image with only black and white
        #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        # Recognize text with tesseract for python
        result = pytesseract.image_to_string(img)
        #print(str(pytesseract.image_to_boxes(img)) + ' result is ' + str(pytesseract.image_to_data(img)))
        #if result=='':
        #    return 'No text'
        #else:
        return str(result)
