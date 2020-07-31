import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)

# cv2.line(img, (0, 0), (512//2, 512//2), (0, 255, 0), 3)
cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 3)
cv2.rectangle(img, (0, 0), (512//2, 512//2), (0, 0, 255), 3)
cv2.circle(img, (512//2//2, 512//2//2), 512//2//2, (255, 0, 0), 3)
cv2.putText(img, "OpenCV", (0, 512//2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 3)

cv2.imshow('Image', img)
cv2.waitKey(0)