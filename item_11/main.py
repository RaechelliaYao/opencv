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


img = cv2.imread('./Resources/ocr_a_reference.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_dilate = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
contours = sorted(contours, key=lambda x:cv2.boundingRect(x)[0])

digits = {}
for i, cnt in enumerate(contours):
    # cv2.drawContours(img, cnt, -1, (0, 0, 255), 3)
    (x, y, w, h) = cv2.boundingRect(cnt)
    roi = img_dilate[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

img = cv2.imread('./Resources/credit_card_01.png')
img_contour = img
# print(img.shape[0], img.shape[1])
img = cv2.resize(img, (300, 300*img.shape[0]//img.shape[1]))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, rectKernel)
img_sobel = cv2.Sobel(img_tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
img_sobel = np.absolute(img_sobel)
val_min, val_max = np.min(img_sobel), np.max(img_sobel)
img_sobel = (255 * ((img_sobel - val_min) / (val_max - val_min))).astype('uint8')

img_dilate_1 = cv2.morphologyEx(img_sobel, cv2.MORPH_CLOSE, rectKernel)
img_thresh_1 = cv2.threshold(img_dilate_1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
img_dilate_2 = cv2.morphologyEx(img_thresh_1, cv2.MORPH_CLOSE, sqKernel)

contours, hierarchy = cv2.findContours(img_dilate_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_contour, contours, -1, (0, 0, 255), 3)
bbox = []
bbox_result = []
# print(len(contours))
for cnt in contours:
    # cv2.drawContours(img_contour, cnt, -1, (0, 0, 255), 3)
    x, y, w, h = cv2.boundingRect(cnt)
    ar = w / float(h)
    if 2.5 < ar < 4.0:
        if (40 < w < 55) and (10 < h < 20):
            bbox.append((x, y, w, h))

bbox = sorted(bbox, key=lambda x: x[0])

for (gX, gY, gW, gH) in bbox:
    output = []
    img_group = img_gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    img_group_thresh = cv2.threshold(img_group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img_group_contours, img_group_hierarchy = cv2.findContours(img_group_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in img_group_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = img_group_thresh[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        scores = []
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        output.append(str(np.argmax(scores)))
    cv2.rectangle(img, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(img, "".join(output[::-1]), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    bbox_result.extend(output)

img_contour = img.copy()

img_stack = stackImages(1, ([img, img_gray, img_tophat],
                              [img_sobel, img_dilate_1, img_thresh_1],
                              [img_dilate_2, img_contour, img_contour]))

cv2.imshow('Stack Image', img_stack)
cv2.waitKey(0)
