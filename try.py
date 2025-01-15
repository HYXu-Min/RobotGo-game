import cv2 as cv
cv.namedWindow('video',cv.WINDOW_AUTOSIZE)
cv.resizeWindow('video',960,640)

cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    cv.imshow('video',frame)

    key = cv.waitKey(1)

    if (key &0xff == ord('q')):

        break

cap.release()
cv.destroyAllWindows()


