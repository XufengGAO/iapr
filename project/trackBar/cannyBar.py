import cv2 as cv
import numpy as np

ori_img = cv.imread(r'C:\Users\24833\Desktop\IAPR\data\train\train_02.jpg')
img = ori_img[2800:3800, 1500:4700]
#img = ori_img

scale_precent =30
width = int(img.shape[1] * scale_precent/100)
height = int(img.shape[0] * scale_precent/100)
dim = (width, height)
img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def nothing():
    pass

cv.namedWindow("bar")
cv.createTrackbar("threshold1", "bar", 0, 255, nothing)
cv.createTrackbar("threshold2", "bar", 0, 255, nothing)

dst = cv.equalizeHist(gray_img)
# 高斯滤波降噪
gaussian = cv.GaussianBlur(dst, (13,13), 0)
cv.imshow("gaussian", gaussian)

while True:
    threshold1 = cv.getTrackbarPos("threshold1", "bar")
    threshold2 = cv.getTrackbarPos("threshold2", "bar")
    # 边缘检测
    edges = cv.Canny(gaussian, threshold1, threshold2)
    cv.imshow("edges", edges)
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()