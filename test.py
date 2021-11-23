import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread(r'D:\SoureDoAn\SDoAn\code\Example\2.png')

alpha = 1.95 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)

manual_result = cv2.convertScaleAbs(image, alpha=(255.0), beta=beta)

cv2.imshow('original', image)
cv2.imshow('manual_result', manual_result)
cv2.waitKey()

