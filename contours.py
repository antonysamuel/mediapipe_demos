import cv2
import numpy as np


img = cv2.imread('assets/fruits.png')

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        r,g,b = img[x,y]
        if r > 200 and g> 200 and b > 200:
            img[x,y] = [255,255,255]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


cv2.drawContours(img, contours, -1, (0,255, 0), 3)
cv2.imshow('img', img)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()