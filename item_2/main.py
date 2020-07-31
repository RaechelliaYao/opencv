import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)

img = cv2.imread('./Resources/lena.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blue = cv2.GaussianBlur(img_gray, (7, 7), 0)
img_canny = cv2.Canny(img, 150, 200)
img_dilate = cv2.dilate(img_canny, kernel, iterations=1)
img_eroded = cv2.erode(img_dilate, kernel, iterations=1)

cv2.imshow('Image', img)
cv2.imshow('Gray Image', img_gray)
cv2.imshow('Blue Image', img_blue)
cv2.imshow('Canny Image', img_canny)
cv2.imshow('Dilate Image', img_dilate)
cv2.imshow('Eroded Image', img_eroded)
cv2.waitKey(0)

