import cv2
import numpy as np
import pytesseract

image = cv2.imread('8.jpg', 1)
# 灰度图片
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 二值化
binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)

rows, cols = binary.shape
scale = 40
# 识别横线
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
eroded = cv2.erode(binary, kernel, iterations=1)
cv2.imshow("Eroded Image",eroded)
dilatedcol = cv2.dilate(eroded, kernel, iterations=1)


# 识别竖线
scale = 20
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
eroded = cv2.erode(binary, kernel, iterations=1)
dilatedrow = cv2.dilate(eroded, kernel, iterations=1)


# 标识交点
bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)
cv2.imshow("表格交点展示：", bitwiseAnd)
cv2.waitKey(0)


# 识别黑白图中的白色交叉点，将横纵坐标取出
ys, xs = np.where(bitwiseAnd > 0)

mylisty = []  # 纵坐标
mylistx = []  # 横坐标

# 通过排序，获取跳变的x和y的值，说明是交点，否则交点会有好多像素值值相近，我只取相近值的最后一点
# 这个10的跳变不是固定的，根据不同的图片会有微调，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
i = 0
myxs = np.sort(xs)
for i in range(len(myxs) - 1):
    if (myxs[i + 1] - myxs[i] > 10):
        mylistx.append(myxs[i])
    i = i + 1
mylistx.append(myxs[i])  # 要将最后一个点加入

i = 0
myys = np.sort(ys)
# print(np.sort(ys))
for i in range(len(myys) - 1):
    if (myys[i + 1] - myys[i] > 10):
        mylisty.append(myys[i])
    i = i + 1
mylisty.append(myys[i])  # 要将最后一个点加入

print('mylisty', mylisty)
print('mylistx', mylistx)
