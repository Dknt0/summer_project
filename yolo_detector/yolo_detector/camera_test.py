#!/usr/bin/python3
import cv2 as cv
import detect


cap = cv.VideoCapture(0)
a = detect.detectapi(weights='yolov5s.pt')
while True:

    rec,img = cap.read()

    result,names =a.detect([img])
    img=result[0][0] #第一张图片的处理结果图片

    for cls,(x1,y1,x2,y2),conf in result[0][1]: #第一张图片的处理结果标签。
        print(cls,x1,y1,x2,y2,conf)
        cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
        cv.putText(img,names[cls],(x1,y1-20),cv.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))

    cv.imshow("vedio",img)

    if cv.waitKey(1)==ord('q'):
        break
