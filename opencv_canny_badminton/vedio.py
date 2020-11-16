import cv2
import numpy as np

def empty(a):
    pass
path = "Resources/badminton5.png"
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,560)
#调节单元名称，调节窗口名称，初始值，最大值
cv2.createTrackbar("Hue Min","TrackBars",0,255,empty)
cv2.createTrackbar("Hue Max","TrackBars",255,255,empty)
cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
    img = cv2.imread(path)
   # img = cv2.resize(img,(760,340))
    kernel1 = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img,kernel1,iterations=1)
    kernel2 = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(erosion,kernel1,iterations=2)
    imgHSV = cv2.cvtColor(dilation,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)#lower与upper之间
    imgResult = cv2.bitwise_and(img,img,mask=mask)


    cv2.imshow("Original",img)
    cv2.imshow("HSV",imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)

    # #imgStack = stackImages(0.6,([img,imgHSV],[mask,imgResult]))
    # cv2.imshow("Stacked Images", imgStack)

    cv2.waitKey(1)

