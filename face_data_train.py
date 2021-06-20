#导入模块
import os
import cv2
from PIL import Image
import sys
import numpy as np

def getImageAndLabels(path):
    facesSamples = []
    ids = []
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    #检测人脸
    face_detrctor = cv2.CascadeClassifier(
        './xml/haarcascade_frontalface_default.xml')
    #遍历列表中的图片
    for imagePaths in imagePaths:
        #打开图片
        PIL_img = Image.open(imagePaths).convert('L')
        #将图像转化为数组
        img_numpy = np.array(PIL_img,'uint8')
        faces = face_detrctor.detectMultiScale(img_numpy)
        #获取每张图片的id
        id = int(os.path.split(imagePaths)[1].split('.')[0])
        print(id)
        for x,y,w,h in faces:
            facesSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    print(imagePaths)
    return facesSamples,ids
if __name__ =='__main__':
    #图片路径
    path = './data'
    #获取图像数组和id数组
    faces,ids = getImageAndLabels(path)
    #获取循环对象
    recong = cv2.face.LBPHFaceRecognizer_create()
    recong.train(faces,np.array(ids))
    #保存文件
    recong.write('./train/trainer.yml')
    print("训练已经完成\n")