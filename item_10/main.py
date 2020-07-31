import cv2
import numpy as np

frame_width = 640
frame_height = 480

cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)
cap.set(10, 150)
color = []


def find_color(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array()
    upper = np.array()
    img_mask = cv2.inRange(img_hsv, lower, upper)
    cv2.imshow('Mask Image', img_mask)


while True:
    success, img = cap.read()
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
