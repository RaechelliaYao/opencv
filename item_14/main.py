import cv2
import numpy as np


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
img = cv2.imread('./Resources/test_01.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blue = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_canny = cv2.Canny(img_blue, 75, 200)

contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

if len(contours) > 0:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4:
            bbox = approx
            break

bbox = bbox.reshape((4, 2))
rect = np.zeros((4, 2), dtype="float32")
s = bbox.sum(axis=1)
rect[0] = bbox[np.argmin(s)]
rect[2] = bbox[np.argmax(s)]
diff = np.diff(bbox, axis=1)
rect[1] = bbox[np.argmin(diff)]
rect[3] = bbox[np.argmax(diff)]
(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))
dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")
M = cv2.getPerspectiveTransform(rect, dst)
img_warped = cv2.warpPerspective(img_gray, M, (maxWidth, maxHeight))


img_thresh = cv2.threshold(img_warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
warped_contours, warped_hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_warped = cv2.cvtColor(img_warped, cv2.COLOR_GRAY2BGR)
# cv2.drawContours(img_warped, warped_contours, -1, (0, 0, 255), 3)
question_cnts = []
for cnt in warped_contours:
    (x, y, w, h) = cv2.boundingRect(cnt)
    ar = w / float(h)
    if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
        question_cnts.append(cnt)

question_cnts = sort_contours(question_cnts, method="top-to-bottom")[0]

correct = 0
for (q, i) in enumerate(np.arange(0, len(question_cnts), 5)):
    cnts = sort_contours(question_cnts[i:i + 5])[0]
    bubbled = None
    for (j, c) in enumerate(cnts):
        img_mask = np.zeros(img_thresh.shape, dtype='uint8')
        cv2.drawContours(img_mask, [c], -1, 255, -1)
        img_bit = cv2.bitwise_and(img_thresh, img_thresh, mask=img_mask)
        total = cv2.countNonZero(img_bit)
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)
    color = (0, 0, 255)
    k = ANSWER_KEY[q]
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1
    cv2.drawContours(img_warped, [cnts[k]], -1, color, 3)

score = (correct / 5.0) * 100
cv2.putText(img_warped, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
img_stack = stackImages(0.7, ([img, img_gray, img_blue, img_canny, img_warped]))
cv2.imshow('Stack Image', img_stack)
cv2.waitKey(0)