import cv2
import numpy as np

img = cv2.imread("resources/u=3029279063,4116942643&fm=26&gp=0.jpg")
img = cv2.resize(img,(640,560))

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def getContours(img):
    # RETR_EXTERNAL:检索极端外部轮廓
    # 第一个参数是寻找轮廓的图像,第二个参数表示轮廓的检索模式,第三个参数method为轮廓的近似办法,contours：返回的轮廓,hierarchy：每条轮廓对应的属性
    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_EXTERNAL,
                                           method= cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 90:
            # 画出轮廓
            # (载体,轮廓,-1(绘制所有的轮廓就用-1),颜色,厚度)
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            # 计算轮廓的长度，图形封闭为true？
            peri = cv2.arcLength(cnt, True)

            # 计算近似拐角点，第二个参数表示的是精度，越小精度越高，因为表示的意思是原始曲线与近似曲线之间的最大距离
            approx = cv2.approxPolyDP(cnt, 0.009 * peri, True)
            print(len(approx))
            objCor = len(approx)

            # 创建边界框,x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
            x, y, w, h = cv2.boundingRect(approx)

            if objCor == 3:
                objectType = "Tri"
            elif objCor == 4:
                aspRatio = w / float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            else:
                objectType = "Circle"
                #原图，左上角，右下角，颜色，宽度
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 标注
            # (载体,类型,位置,字体,比例,颜色,厚度)
            cv2.putText(imgContour, objectType,
                        (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX,
                        0.9, (0, 0, 0), 2)


imgContour = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)
imgBlank = np.zeros_like(img)

getContours(imgCanny)

imgStack = stackImages(0.65, ([img, imgGray, imgBlur],
                              [imgCanny, imgContour, imgBlank]))

cv2.imshow("Stack", imgStack)
cv2.waitKey(0)

