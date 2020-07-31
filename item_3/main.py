import cv2

img = cv2.imread('./Resources/lambo.PNG')
print(img.shape)
img_resize = cv2.resize(img, (1000, 500))
img_cropped = img[150:350, 350:550]

cv2.imshow('Image', img)
cv2.imshow('Resize Image', img_resize)
cv2.imshow('Cropped Image', img_cropped)
cv2.waitKey(0)

