import cv2
import matplotlib.pyplot as plt
import numpy as np

path = '0.jpg'
img = cv2.imread(path)
imgContour = img.copy()

# 从一幅Harris响应图像中返回角点，min_dist为分割角点和图像边界的最少像素数目
def getContours(img):
    # 查找轮廓，cv2.RETR_ExTERNAL=获取外部轮廓点, CHAIN_APPROX_NONE = 得到所有的像素点
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 循环轮廓，判断每一个形状
    for cnt in contours:
        # 获取轮廓面积
        area = cv2.contourArea(cnt)
        # 当面积大于500，代表有形状存在
        if area > 500:
            # 绘制所有的轮廓并显示出来
            cv2.drawContours(imgContour, cnt, -1, (0, 0, 0), 3)
            # 计算所有轮廓的周长，便于做多边形拟合
            peri = cv2.arcLength(cnt, True)
            # 多边形拟合，获取每个形状的边数目
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            # 获取每个形状的左上角坐标xy和形状的长宽wh
            x, y, w, h = cv2.boundingRect(approx)
            # 计算出边界后，即边数代表形状，如三角形边数=3
            if objCor == 3:
                objectType = "Tri"
            # 计算出边界后，即边数代表形状，如四边形边数=4
            elif objCor == 4:
                # 判断是矩形还是正方形
                aspRatio = w / float(h)
                if aspRatio > 0.98 and aspRatio < 1.03:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"

            # 绘制文本时需要绘制在图形附件
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return x, y ,  x + w, y + h

# 灰度化
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 高斯平滑
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
# 边缘检测
imgCanny = cv2.Canny(imgBlur, 50, 50)
# 获取轮廓特征点
minx,miny,maxx,maxy = getContours(imgCanny)
plt.figure(figsize=(30, 20));

plt.subplot(235);plt.imshow(imgContour);plt.title('imgContour');plt.axis('off');plt.show();


lensx=np.uint16(np.around((maxx-minx)/8))
lensy=np.uint16(np.around((maxy-miny)/8))

qipan = cv2.imread('0.jpg',1)
qipan = cv2.cvtColor(qipan, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(qipan, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,200,apertureSize = 3)

print(minx,maxx,miny,maxy,lensx,lensy)
circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=24, minRadius=15, maxRadius=25)
circles = circle1[0, :, :]  # 提取为二维
circles = np.uint16(np.around(circles))  # 四舍五入，取整
print(circles)
circles = np.array(circles) #
for i in circles[:]:
    # cv2.circle(qipan, (i[0], i[1]), i[2], (255, 0, 0), 5)  # 画圆
    cv2.circle(qipan, (i[0], i[1]), 1, (255, 0, 0), 10)  # 画圆心
plt.imshow(qipan)
plt.show()



pan=np.zeros((9,9),int)
for n in range(len(circles[:,0])):

    x=np.uint16(np.around((circles[n,0]-minx)/lensx))

    y=np.uint16(np.around((circles[n,1]-miny)/lensy))

    pan[y,x]=1





print(pan)




