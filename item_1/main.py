import cv2

# Read Image
# img = cv2.imread('./Resources/lena.png')
# cv2.imshow('Lena Image', img)
# cv2.waitKey(0)


# Read Video
# frame_width = 640
# frame_height = 480
# cap = cv2.VideoCapture('./Resources/test_video.mp4')
# while True:
#     success, img = cap.read()
#     img = cv2.resize(img, (frame_width, frame_height))
#     cv2.imshow('Test Video', img)
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break


# Read Camera
frame_width = 640
frame_height = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)
while True:
    success, img = cap.read()
    cv2.imshow('Online Camera', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break