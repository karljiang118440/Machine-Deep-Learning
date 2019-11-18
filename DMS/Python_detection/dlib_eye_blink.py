#coding=utf-8  
import numpy as np 
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils

pwd = os.getcwd()# 获取当前路径
model_path = os.path.join(pwd, 'model')# 模型文件夹路径
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')# 人脸特征点检测模型路径

detector = dlib.get_frontal_face_detector()# 人脸检测器
predictor = dlib.shape_predictor(shape_detector_path)# 人脸特征点检测器

EYE_AR_THRESH = 0.3# EAR阈值
EYE_AR_CONSEC_FRAMES = 3# 当EAR小于阈值时，接连多少帧一定发生眨眼动作

#判断闭眼时间的帧数：EYE_AR_CLOSE_CONSEC_FRAMES
EYE_AR_CLOSE_CONSEC_FRAMES = 60# 当EAR小于阈值时，接连多少帧一定发生眨眼动作,并且制定

# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

frame_counter = 0# 连续帧计数 
blink_counter = 0# 眨眼计数
cap = cv2.VideoCapture(1)
while(1):
    ret, img = cap.read()# 读取视频流的一帧

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 转成灰度图像
    rects = detector(gray, 0)# 人脸检测
    for rect in rects:# 遍历每一个人脸
        print('-'*20)
        shape = predictor(gray, rect)# 检测特征点
        points = face_utils.shape_to_np(shape)# convert the facial landmark (x, y)-coordinates to a NumPy array
        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]# 取出左眼对应的特征点
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]# 取出右眼对应的特征点
        leftEAR = eye_aspect_ratio(leftEye)# 计算左眼EAR
        rightEAR = eye_aspect_ratio(rightEye)# 计算右眼EAR
        print('leftEAR = {0}'.format(leftEAR))
        print('rightEAR = {0}'.format(rightEAR))

        ear = (leftEAR + rightEAR) / 2.0# 求左右眼EAR的均值

        leftEyeHull = cv2.convexHull(leftEye)# 寻找左眼轮廓
        rightEyeHull = cv2.convexHull(rightEye)# 寻找右眼轮廓
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)# 绘制左眼轮廓
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)# 绘制右眼轮廓

        # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
        if ear < EYE_AR_THRESH:
            frame_counter += 1
        else:
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                blink_counter += 1
            frame_counter = 0

        #如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CLOSE_CONSEC_FRAMES时，才会计做一次连续闭眼
            ##karl:2019.11.08
        if ear < EYE_AR_THRESH:
            frame_counter_closed += 1
        else:
            if frame_counter >= EYE_AR_CLOSE_CONSEC_FRAMES:
                print("your eye clsed , too dangeous")
            frame_counter_closed = 0

           

        # 在图像上显示出眨眼次数blink_counter和EAR
        cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)




#####################  依据测量眼测量嘴巴打哈欠状态 ######################################################
#karl:2019.11.08


        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]# 取出左眼对应的特征点
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]# 取出右眼对应的特征点
        leftEAR = eye_aspect_ratio(leftEye)# 计算左眼EAR
        rightEAR = eye_aspect_ratio(rightEye)# 计算右眼EAR
        print('leftEAR = {0}'.format(leftEAR))
        print('rightEAR = {0}'.format(rightEAR))

        ear = (leftEAR + rightEAR) / 2.0# 求左右眼EAR的均值

        leftEyeHull = cv2.convexHull(leftEye)# 寻找左眼轮廓
        rightEyeHull = cv2.convexHull(rightEye)# 寻找右眼轮廓
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)# 绘制左眼轮廓
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)# 绘制右眼轮廓

        # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
        if ear < EYE_AR_THRESH:
            frame_counter += 1
        else:
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                blink_counter += 1
            frame_counter = 0

        #如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CLOSE_CONSEC_FRAMES时，才会计做一次连续闭眼
            ##karl:2019.11.08
        if ear < EYE_AR_THRESH:
            frame_counter_closed += 1
        else:
            if frame_counter >= EYE_AR_CLOSE_CONSEC_FRAMES:
                print("your eye clsed , too dangeous")
            frame_counter_closed = 0

           

        # 在图像上显示出眨眼次数blink_counter和EAR
        cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)












    cv2.imshow("Frame", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
