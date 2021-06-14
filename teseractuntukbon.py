import pytesseract
from PIL import Image
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
file_path= 'C://Users//ASUS//PycharmProjects//pythonProject1//tes.png'
#file_path = 'C://Users//ASUS//PycharmProjects//pythonProject1//Plate//14.jpg'
im = Image.open(file_path)
im.save('ocr.png', dpi=(300, 300))

image = cv2.imread('ocr.png')
image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
retval, threshold = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

text = pytesseract.image_to_string(threshold)
print(text)
