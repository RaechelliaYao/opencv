import cv2
import numpy as np


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


img = cv2.imread('./Resources/page.jpg')
img = cv2.resize(img, (500*img.shape[1]//img.shape[0], 500))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blue = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_canny = cv2.Canny(img_blue, 75, 200)

contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
# print(contours)
for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
    if len(approx) == 4:
        screen_cnt = approx
        break

print(screen_cnt)
print(screen_cnt.shape)

cv2.drawContours(img, [screen_cnt], -1, (0, 0, 255), 2)
# screen_cnt = screen_cnt.reshape(4, 2)*img.shape[0]//500
screen_cnt = screen_cnt.reshape(4, 2)
rect = np.zeros((4, 2), dtype='float32')
screen_cnt_sum = screen_cnt.sum(axis=1)
screen_cnt_diff = np.diff(screen_cnt, axis=1)
rect[0] = screen_cnt[np.argmin(screen_cnt_sum)]
rect[1] = screen_cnt[np.argmin(screen_cnt_diff)]
rect[2] = screen_cnt[np.argmax(screen_cnt_sum)]
rect[3] = screen_cnt[np.argmax(screen_cnt_diff)]
(tl, tr, br, bl) = rect

widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")

M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
img_dilate = cv2.threshold(warped_gray, 100, 255, cv2.THRESH_BINARY)[1]

img_stack = stackImages(1, ([img, img_gray, img_blue, img_canny]))
cv2.imshow('Dilate Image', img_dilate)
cv2.imshow('Stack Image', img_stack)
cv2.waitKey(0)