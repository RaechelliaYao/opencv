import cv2
import numpy as np
import operator


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


class_dictionary = {1: 'empty', 2: 'occupied'}
img = cv2.imread('./Resources/scene1380.jpg')

lower = np.uint8([120, 120, 120])
upper = np.uint8([255, 255, 255])
img_mask = cv2.inRange(img, lower, upper)
img_bit = cv2.bitwise_and(img, img, mask=img_mask)
img_gray = cv2.cvtColor(img_bit, cv2.COLOR_BGR2GRAY)
img_canny = cv2.Canny(img, 50, 200)

rows, cols = img.shape[:2]
pt_1 = [cols*0.05, rows*0.90]
pt_2 = [cols*0.05, rows*0.70]
pt_3 = [cols*0.30, rows*0.55]
pt_4 = [cols*0.6, rows*0.15]
pt_5 = [cols*0.90, rows*0.15]
pt_6 = [cols*0.90, rows*0.90]
vertices = np.array([[pt_1, pt_2, pt_3, pt_4, pt_5, pt_6]], dtype=np.int32)
img_point = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
for point in vertices[0]:
    cv2.circle(img_point, (point[0], point[1]), 10, (0, 0, 255), 4)

# print(len(img_canny.shape))
img_zero = np.zeros_like(img_canny)
cv2.fillPoly(img_zero, vertices, 255)
img_point_bit = cv2.bitwise_and(img_canny, img_zero)

img_lines = cv2.HoughLinesP(img_point_bit, rho=0.1, theta=np.pi/10, threshold=15, minLineLength=9, maxLineGap=4)
print(img_lines.shape)
cleaned = []
for line in img_lines:
    for x1, y1, x2, y2 in line:
        if 1 >= abs(y2 - y1) and 25 <= abs(x2 - x1) <= 55:
            cleaned.append((x1, y1, x2, y2))
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
cleaned_list = sorted(cleaned, key=operator.itemgetter(0, 1))
clusters = {}
dIndex = 0
for i in range(len(cleaned_list) - 1):
    distance = abs(cleaned_list[i + 1][0] - cleaned_list[i][0])
    if distance <= 10:
        if not dIndex in clusters.keys(): clusters[dIndex] = []
        clusters[dIndex].append(cleaned_list[i])
        clusters[dIndex].append(cleaned_list[i + 1])
    else:
        dIndex += 1
print(clusters)
rects = {}
i = 0
for key in clusters:
    all_list = clusters[key]
    cleaned = list(set(all_list))
    if len(cleaned) > 5:
        cleaned = sorted(cleaned, key=lambda x: x[1])
        avg_y1 = cleaned[0][1]
        avg_y2 = cleaned[-1][1]
        avg_x1 = 0
        avg_x2 = 0
        for tup in cleaned:
            avg_x1 += tup[0]
            avg_x2 += tup[2]
        avg_x1 = avg_x1/len(cleaned)
        avg_x2 = avg_x2/len(cleaned)
        rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
        i += 1
buff = 7
for key in rects:
    tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
    tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
    cv2.rectangle(img, tup_topLeft, tup_botRight, (0, 255, 0), 3)

print(rects)

# gap = 15.5
# spot_dict = {}
# tot_spots = 0
# adj_y1 = {0: 20, 1: -10, 2: 0, 3: -11, 4: 28, 5: 5, 6: -15, 7: -15, 8: -10, 9: -30, 10: 9, 11: -32}
# adj_y2 = {0: 30, 1: 50, 2: 15, 3: 10, 4: -15, 5: 15, 6: 15, 7: -20, 8: 15, 9: 15, 10: 0, 11: 30}
# adj_x1 = {0: -8, 1: -15, 2: -15, 3: -15, 4: -15, 5: -15, 6: -15, 7: -15, 8: -10, 9: -10, 10: -10, 11: 0}
# adj_x2 = {0: 0, 1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 10, 9: 10, 10: 10, 11: 0}
# for key in rects:
#     tup = rects[key]
#     x1 = int(tup[0] + adj_x1[key])
#     x2 = int(tup[2] + adj_x2[key])
#     y1 = int(tup[1] + adj_y1[key])
#     y2 = int(tup[3] + adj_y2[key])
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     num_splits = int(abs(y2 - y1) // gap)
#     for i in range(0, num_splits + 1):
#         y = int(y1 + i * gap)
#         cv2.line(img, (x1, y), (x2, y), (255, 0, 0), 2)
#     if 0 < key < len(rects) - 1:
#         x = int((x1 + x2) / 2)
#         cv2.line(img, (x, y1), (x, y2), (255, 0, 0), 2)
#
#     if key == 0 or key == (len(rects) - 1):
#         tot_spots += num_splits + 1
#     else:
#         tot_spots += 2 * (num_splits + 1)
#
#     if key == 0 or key == (len(rects) - 1):
#         for i in range(0, num_splits + 1):
#             cur_len = len(spot_dict)
#             y = int(y1 + i * gap)
#             spot_dict[(x1, y, x2, y + gap)] = cur_len + 1
#     else:
#         for i in range(0, num_splits + 1):
#             cur_len = len(spot_dict)
#             y = int(y1 + i * gap)
#             x = int((x1 + x2) / 2)
#             spot_dict[(x1, y, x, y + gap)] = cur_len + 1
#             spot_dict[(x, y, x2, y + gap)] = cur_len + 2

img_stack = stackImages(0.5, ([img, img_mask, img_bit], [img_gray, img_canny, img_point], [img_zero, img_point_bit, img_point_bit]))
cv2.imshow('Stack Image', img_stack)

cv2.imshow('Image', img)
# cv2.imshow('Mask Image', img_mask)
# cv2.imshow('Bit Image', img_bit)
# cv2.imshow('Gray Image', img_gray)
# cv2.imshow('Canny Image', img_canny)
# cv2.imshow('Point Image', img_point)
# cv2.imshow('Line Image', img_line)
# cv2.imshow('Zero Image', img_zero)
# cv2.imshow('Point Bit Image', img_point_bit)
cv2.waitKey(0)