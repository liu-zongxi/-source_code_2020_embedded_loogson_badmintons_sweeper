import cv2
import numpy as np
###########################################################
img = cv2.imread("resources/test9.jpg")#读取存储路径
img = cv2.resize(img,(500,350))
cv2.imwrite("resources/copy1.jpg", img)
imgContour = img.copy()         # 生成一个原图，否则框会画在mask上
#################################################################
# frameWidth = 640
# frameHeight = 480
# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10,150)

#######################################################
lower = np.array([0,0,126])
upper = np.array([68,85,250])
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #要处理的图像转换成HSV
mask = cv2.inRange(imgHSV,lower, upper)         #生成mask

def getContours(img):
    # RETR_EXTERNAL:检索极端外部轮廓
    # 第一个参数是寻找轮廓的图像,第二个参数表示轮廓的检索模式,第三个参数method为轮廓的近似办法,contours：返回的轮廓,hierarchy：每条轮廓对应的属性
    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 8:
            # 画出轮廓
            # (载体,轮廓,-1(绘制所有的轮廓就用-1),颜色,厚度)
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            # # 计算轮廓的长度，图形封闭为true？
            peri = cv2.arcLength(cnt, True)

            # 计算近似拐角点，第二个参数表示的是精度，越小精度越高，因为表示的意思是原始曲线与近似曲线之间的最大距离
            approx = cv2.approxPolyDP(cnt, 0.009 * peri, True)
            print(len(approx))
            objCor = len(approx)

            # 创建边界框,x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
            x, y, w, h = cv2.boundingRect(approx)

            # if objCor == 4:
            #     objectType = "Badminton"
            # elif objCor == 4:
            #     aspRatio = w / float(h)
            #     if aspRatio > 0.95 and aspRatio < 1.05:
            #         objectType = "Square"
            #     else:
            #         objectType = "Rectangle"
            # else:
            #     objectType = "Else"
                #原图，左上角，右下角，颜色，宽度
            cv2.rectangle(imgContour, (x - 10, y - 10), (x + w + 30, y + h +10), (0, 255, 0), 2)

            # 标注
            # (载体,类型,位置,字体,比例,颜色,厚度)
            cv2.putText(imgContour, "badminton",
                        (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX,
                        0.9, (0, 255, 0), 2)

imgMask = cv2.bitwise_and(img,img,mask=mask)        #把img和mask做and操作，使得图片二值化
cv2.imshow("imgmask",imgMask)
imgblur1 = cv2.GaussianBlur(imgMask,(13,13),10,10)      #做两次高斯滤波，边缘平滑（查）
imgblur2 = cv2.GaussianBlur(imgblur1,(7,7),30,30)
kernel = np.ones((3,3),np.uint8)                        #膨胀，查
imgdialation = cv2.dilate(imgblur2, kernel, iterations=2)
imgCanny = cv2.Canny(imgdialation, 80, 120)             # 寻找边界
getContours(imgCanny)                                   # 画框
cv2.imshow("canny",imgCanny)
cv2.imshow("Result",imgContour)

cv2.imwrite("resources/result1.jpg", imgContour)
cv2.imwrite("resources/canny1.jpg", imgCanny)
cv2.imwrite("resources/mask1.jpg", imgMask)
cv2.waitKey(0)
####################################################
# while True:
#     success, img = cap.read()
#     imgContour = img.copy()
#     lower = np.array([0, 0, 227])
#     upper = np.array([37, 58, 255])
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 要处理的图像转换成HSV
#     mask = cv2.inRange(imgHSV, lower, upper)
#     imgMask = cv2.bitwise_and(img, img, mask=mask)  # 把img和mask做and操作，使得图片二值化
#     cv2.imshow("imgmask", imgMask)
#     imgblur1 = cv2.GaussianBlur(imgMask, (13, 13), 10, 10)  # 做两次高斯滤波，边缘平滑（查）
#     imgblur2 = cv2.GaussianBlur(imgblur1, (7, 7), 30, 30)
#     kernel = np.ones((3, 3), np.uint8)  # 膨胀，查
#     imgdialation = cv2.dilate(imgblur2, kernel, iterations=2)
#     imgCanny = cv2.Canny(imgdialation, 80, 120)  # 寻找边界
#     getContours(imgCanny)
#
#     cv2.imshow("Result", imgContour)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
