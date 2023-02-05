#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 0. 导入模块
import os
import glob
import numpy as np
import cv2
import dlib
import sys

# 1. 加载模型、图片

## 正向人脸检测器
detector = dlib.get_frontal_face_detector()
## 加载特征点提取模型
predictor_path = 'shape_predictor_5_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
## 加载面部识别模型
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
## 已知图片
known_image_path = "image/known"
## 测试图片
test_image_path = "image/unknown"


# In[2]:


# 2. 声明存放数据的变量

## 声明descriptors，用于存放已知图片对应的人脸特征向量
descriptors = []
## 声明names，用于存放于人脸特征向量对应的名字。
names = ["huge",'oly','pyy']


# In[3]:


# 3. 开始检测

## 遍历known_image_path文件夹下所有以.png结尾的文件。
for f in glob.glob(os.path.join(known_image_path, "*.png")):
    print(f)
    img = dlib.load_rgb_image(f)
    ## 使用 detector 检测器来检测图像中的人脸
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        ## 获取人脸特征点
        shape = predictor(img, d)
        ## 计算特征向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        ## 特征向量转换为numpy array
        v = np.array(face_descriptor)
        ## 把此次数据存到人脸特征向量列表里面
        descriptors.append(v)


# In[6]:


# 获取当前人脸并进行判断

#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0)  

def face_rec(img):
    # 5 建立窗口
    dets = detector(img, 1)  # 检测 1是放大图片

    for k, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0

        cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)

        c = cv2.waitKey(50)
        if c == 27:
            cv2.destroyAllWindows()
            cam.release()
            sys.exit(0)

        ## 获取人脸特征点
        shape = predictor(img, d)
        ## 计算特征向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        ## 将当前待判断的图片特征向量转化为 current
        current = np.array(face_descriptor)

        ## 计算欧式距离，识别人脸
        ### 设置阈值
        tolerance = 0.5
        ### 设置该图片名字初值为：Unknow
        current_name = "Unknow name"
        ### 对这个存放着已知图片特征向量的列表descriptors[]遍历
        for i in range(len(descriptors)):
            ### 计算欧氏距离
            distance = np.linalg.norm(descriptors[i] - current)
            ### 对欧氏距离判断
            if distance < tolerance:
                ### names用于存放着人脸特征向量对应的名字
                current_name = names[i]
                break
        ## 输出对当前图片的识别结果


        cv2.putText(img, current_name, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
        cv2.imshow('image', img)
        return current_name
while(True):
    _, img = cam.read()
    current_name = face_rec(img)
    c = cv2.waitKey(50)
    if c == 27:
        cv2.destroyAllWindows()
        cam.release()
        sys.exit(0)
    if current_name:
        print("当前图片识别结果为：" + current_name)


