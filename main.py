import cv2
import numpy as np

cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('thispersondoesnotexist.jpg')
img = cv2.resize(img, (500, 500))
copy = img.copy()
gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
eyes = cascade.detectMultiScale(gray, 1.3, 5)
for (ex, ey, ew, eh) in eyes:
    cv2.rectangle(copy, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


# cv2.imshow('Original', img)
# cv2.imshow('Eyes Detected', copy)
stack = np.hstack([img, copy])
cv2.imshow('Output', stack)
cv2.waitKey(0)