import cv2
import matplotlib.pyplot as plt




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
            # 大于4个边的就是圆形
            elif objCor > 4:
                objectType = "Circles"
            else:
                objectType = "None"
            # 绘制文本时需要绘制在图形附件
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imgContour, objectType,
                        (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 0, 0), 2)

path = 'calibresult.jpg'
img = cv2.imread(path)
imgContour = img.copy()

# 灰度化
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 高斯平滑
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
# 边缘检测
imgCanny = cv2.Canny(imgBlur, 50, 50)
# 获取轮廓特征点
getContours(imgCanny)

plt.figure(figsize=(30, 20));
plt.subplot(231);plt.imshow(img);plt.title('img');plt.axis('off');
plt.subplot(232);plt.imshow(imgGray,cmap='gray');plt.title('imgGray');plt.axis('off');
plt.show()
plt.subplot(233);plt.imshow(imgBlur,cmap='gray');plt.title('imgBlur');plt.axis('off');
plt.subplot(234);plt.imshow(imgCanny,cmap='gray');plt.title('imgCanny');plt.axis('off');
plt.subplot(235);plt.imshow(imgContour);plt.title('imgContour');plt.axis('off');
plt.show()
