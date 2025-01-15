import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

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


qipan = cv.imread('board.jpg',1)
qipan = cv.cvtColor(qipan, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(qipan, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,100,200,apertureSize = 3)
plt.figure(figsize=(30, 20));
plt.subplot(121), plt.imshow(qipan)
plt.title('img'), plt.xticks([]), plt.yticks([])
circle1 = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=100, param2=24, minRadius=5, maxRadius=25)
circles = circle1[0, :, :]  # 提取为二维
circles = np.uint16(np.around(circles))  # 四舍五入，取整
for i in circles[:]:
    cv.circle(qipan, (i[0], i[1]), i[2], (255, 0, 0), 5)  # 画圆
#     cv2.circle(qipan, (i[0], i[1]), 1, (255, 0, 0), 10)  # 画圆心
plt.subplot(122), plt.imshow(qipan)
plt.title('circle'), plt.xticks([]), plt.yticks([]);

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_c1=cv2.imread('board.jpg',0)
img_c1_CC=cv2.imread('board.jpg',1)
img_c1_CC=cv2.cvtColor(img_c1_CC,cv2.COLOR_BGR2RGB)
Harris_c1 = cv2.cornerHarris(img_c1, 3, 3, 0.04)
dst_c1 = cv2.dilate(Harris_c1, None)  #将可以标出来的点粗化
plt.figure(figsize=(10, 10));
img_c1_C=img_c1_CC.copy()
thres = 0.1*dst_c1.max()
img_c1_C[dst_c1 > thres] = [255,0,0]
plt.imshow(img_c1_C)

# 从一幅Harris响应图像中返回角点，min_dist为分割角点和图像边界的最少像素数目
def get_harris_points(harrisim,min_dist=10,threshold=0.1):
    # 寻找高于阈值的候选角点
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    # 得到候选点的坐标
    coords = array(harrisim_t.nonzero()).T
    # 以及它们的 Harris 响应值
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    # 对候选点按照 Harris 响应值进行排序
    index = argsort(candidate_values)[::-1]
    # 将可行点的位置保存到数组中
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
    # 按照 min_distance 原则，选择最佳 Harris 点
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                        (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    return filtered_coords

from pylab import *
from numpy import *
wid=9   #比较像素点数目
filtered_coords1 = get_harris_points(dst_c1, wid+1,0.1)   #图1大于阈值的坐标

maxx=0
for i in range(len(filtered_coords1)):
    if filtered_coords1[i][0]>maxx:
        maxx=filtered_coords1[i][0]
maxy=0
for i in range(len(filtered_coords1)):
    if filtered_coords1[i][1]>maxy:
        maxy=filtered_coords1[i][1]
minx=maxx
for i in range(len(filtered_coords1)):
    if filtered_coords1[i][0]<minx:
        minx=filtered_coords1[i][0]
miny=maxy
for i in range(len(filtered_coords1)):
    if filtered_coords1[i][1]<miny:
        miny=filtered_coords1[i][1]
lensx=np.uint16(np.around((maxx-minx)/18))
lensy=np.uint16(np.around((maxy-miny)/18))

pan=np.zeros((19,19),int)
for n in range(len(circles[:,0])):
    x=np.uint16(np.around((circles[n,0]-minx)/lensx))
    y=np.uint16(np.around((circles[n,1]-miny)/lensy))
    pan[y,x]=1
pan

