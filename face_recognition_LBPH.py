#导入模块
import cv2
import numpy as np
import os

#加载训练数据集文件
recong = cv2.face.LBPHFaceRecognizer_create()
recong.read('./train/trainer.yml')
#准备识别的图片
os.chdir('./img')
img = cv2.imread('444.jpg')
#将图片灰度处理
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#加载特征数据集
face_detrctor = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')
faces = face_detrctor.detectMultiScale(grey)
for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,255,255),thickness=2)
    #人脸识别
    id,confident = recong.predict(grey[y:y+h,x:x+w])
    print('标签：',id,'相似度：',confident)
    if confident < 130:
        print("您不是本人\n")
    else:
        print("您是本人\n")
cv2.imshow('result',img)

while True:
    ord('q') == cv2.waitKey(0)
    break
cv2.destroyAllWindows()