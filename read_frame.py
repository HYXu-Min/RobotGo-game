import time
import cv2
import numpy as np
import glob

# 找棋盘格角点标定并且写入文件
# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 阈值
# 棋盘格模板规格
w = 9   # 9 - 1
h = 9   # 7  - 1
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objp = objp * 21  # 棋盘方块边长21 mm

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点

images = glob.glob('D:/camera_data/*.jpg')  # 拍摄的十几张棋盘图片所在目录

i = 1
for fname in images:
    img = cv2.imread(fname)
    # 获取画面中心点
    h1, w1 = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    u, v = img.shape[:2]
    print(u, v)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        print("i:", i)
        i = i+1
        # 对检测到的角点作进一步的优化计算，可使角点的精度达到亚像素级别
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 640, 480)
        cv2.imshow('findCorners', img)
        cv2.waitKey(200)
cv2.destroyAllWindows()
#  标定
print('正在计算')
ret, mtx, dist, rvecs, tvecs = \
    cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# cv_file = cv2.FileStorage("E:/code/1_21mm_2/camera.yaml", cv2.FILE_STORAGE_WRITE)
# cv_file.write("camera_matrix", mtx)
# cv_file.write("dist_coeff", dist)
# # 请注意，*释放*不会关闭（）FileStorage对象
#
# cv_file.release()

print("ret:", ret)
print("mtx:\n", mtx)      # 内参数矩阵
print("dist畸变值:\n", dist)   # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs旋转（向量）外参:\n", rvecs)   # 旋转向量  # 外参数
print("tvecs平移（向量）外参:\n", tvecs)  # 平移向量  # 外参数
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
print('newcameramtx外参', newcameramtx)

def get_pieces(circles, minx, miny, lensx, lensy,qipan):
    pan = np.zeros((9, 9), int)

    # black pieces
    for n in range(len(circles[:, 0])):
        x = np.uint16(np.around((circles[n, 0] - minx) / lensx))
        y = np.uint16(np.around((circles[n, 1] - miny) / lensy))
        pan[y, x] = 1
    # white pieces
    for n in range(len(circles[:, 0])):
        j = circles[n, 0]
        i = circles[n, 1]
        x = np.uint16(np.around((circles[n, 0] - minx) / lensx))
        y = np.uint16(np.around((circles[n, 1] - miny) / lensy))
        avg = (int(qipan[i, j, 0]) + int(qipan[i, j, 1]) + int(qipan[i, j, 2])) / 3;
        if qipan[i, j, 0] > 190:
            pan[y, x] = 2

    return pan

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

cap = cv2.VideoCapture(1)

ret, frame = cap.read()
while ret:
    for index in range(100):
        print('Begin to take pictures..........')
            #resize = cv2.resize(frame, (512,512), interpolation=cv2.INTER_NEAREST)
        ret, frame = cap.read()
        save_path = 'D:/go_board_Data/'
        cv2.imwrite(save_path+'%d.jpg'%(index), frame)

        time.sleep(3)

        img2 = cv2.imread(save_path+'%d.jpg'%(index))
        h, w = img2.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))  # 自由比例参数
        dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
            # 根据前面ROI区域裁剪图片
            # x,y,w,h = roi
            # dst = dst[y:y+h, x:x+w]
        cv2.imwrite(save_path+'%d.jpg'%(index), dst)
        time.sleep(3)

        frame1 = cv2.imread(save_path+'%d.jpg'%(index),1)
        frame1 = cv2.flip(frame1,-1)

        img = cv2.imread(save_path+'0.jpg')
        imgContour = img.copy()

        qipan = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(qipan, cv2.COLOR_BGR2GRAY)


        edges = cv2.Canny(gray, 100, 200, apertureSize=3)

        circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=24, minRadius=15,
                                       maxRadius = 25)
        circles = circle1[0, :, :]  # 提取为二维
        circles = np.uint16(np.around(circles))  # 四舍五入，取整
        for i in circles[:]:
            cv2.circle(qipan, (i[0], i[1]), i[2], (255, 0, 0), 5)  # 画圆


        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 高斯平滑
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
            # 边缘检测
        imgCanny = cv2.Canny(imgBlur, 50, 50)
            # 获取轮廓特征点
        minx, miny, maxx, maxy = getContours(imgCanny)

        lensx = np.uint16(np.around((maxx - minx) / 8))
        lensy = np.uint16(np.around((maxy - miny) / 8))

        pan = get_pieces(circles, minx, miny, lensx, lensy, qipan)

        print(pan)
        cv2.imwrite(save_path + '%d.jpg' % (index), qipan)
        index += 1






cap.release()
cv2.destroyAllWindows()


