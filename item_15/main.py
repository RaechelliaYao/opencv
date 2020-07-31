import cv2

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

trackers = cv2.MultiTracker_create()
cap = cv2.VideoCapture('./Resources/soccer_01.mp4')

while True:
    success, img = cap.read()
    if not success:
        break

    (img_height, img_width) = img.shape[:2]
    set_width = 600
    r = set_width / float(img_width)
    dim = (set_width, int(img_height * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    (_, boxes) = trackers.update(img)
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Video', img)
    key = cv2.waitKey(100) & 0xff
    if key == ord('s'):
        box = cv2.selectROI('Image', img, fromCenter=False, showCrosshair=True)
        trackers = OPENCV_OBJECT_TRACKERS['kcf']
        trackers.add(trackers, img, box)
    elif key == 27:
        break