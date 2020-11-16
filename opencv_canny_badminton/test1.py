import cv2
import  numpy as np
img = cv2.imread("resources\FrederickSound_ROW5437500422_1920x1080.jpg")#读取存储路径
kernel = np.ones((3,3),np.uint8)#返回全是1的矩阵，（（内核大小），处理对象类型，自行查阅）
# imggrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#将图片灰度处理
imgresize = cv2.resize(img, (960,540))#重设大小
imgcropped = imgresize[150:600, 260:800]#裁剪[高度起点：高度结束，宽度起点：宽度结束]
imgblur1 = cv2.GaussianBlur(imgcropped,(7,7),1,0)#(输入图片，内核大小，sigma_x轴方向标准差，sigma_y轴方向标准差),sigma越大，越类似均值模板，平滑处理
#画出边缘，(输入图片imgblur1，弱阈值，强阈值)如果当前梯度值大于给定的maxVal，判断为边界， 如果当前梯度值小于minval则舍弃，
# 如果当前梯度值在给定的最大值和最小值之间，如果其周围的点是边界点，那么当前点保留，否者舍弃
imgcanny = cv2.Canny(imgblur1,10,200)
imgdialation = cv2.dilate(imgcanny, kernel, iterations=1)#边缘厚度增加（处理图片，使用的内核，迭代的次数）
imgeroded = cv2.erode(imgdialation, kernel, iterations=1)#边缘变细
# cv2.imshow("gray image", imggrey)
#cv2.imshow("image", img)
#cv2.imshow("blur image1", imgblur1)
cv2.imshow("canny image", imgcanny)
cv2.imshow("dialation image", imgdialation)
cv2.imshow("erode image", imgeroded)
cv2.waitKey(0)#0表示任意键结束，其余数字为暂留时间

