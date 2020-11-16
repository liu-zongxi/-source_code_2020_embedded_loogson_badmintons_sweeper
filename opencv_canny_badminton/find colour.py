import cv2
import numpy as np

cap = cv2.VideoCapture(0)
def empty(a):
    pass
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",550,560)
#调节单元名称，调节窗口名称，初始值，最大值
cv2.createTrackbar("Hue Min","TrackBars",0,255,empty)
cv2.createTrackbar("Hue Max","TrackBars",255,255,empty)
cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)


while True:
    success, img = cap.read()
    img = cv2.resize(img,(760,560))
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
  #  print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)#lower与upper之间
    imgResult = cv2.bitwise_and(img,img,mask=mask)


  #  cv2.imshow("Original",img)
    #cv2.imshow("HSV",imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)

    # #imgStack = stackImages(0.6,([img,imgHSV],[mask,imgResult]))
    # cv2.imshow("Stacked Images", imgStack)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
