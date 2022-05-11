# coding = utf-8
'''

date : 2022-4-28
desc:  在屏幕上画圆点

'''

import cv2
from cv2 import SOLVEPNP_ITERATIVE
import dlib
import math
import numpy as np
#import os
from imutils import face_utils
import pygame

#import  HPEMain
#import sys


def face_orientation(frame, landmarks):
    size = frame.shape  # (height, width, color_channel)

    image_points = np.array([

        #源文件
        landmarks[33],  # Nose tip
        landmarks[8],   # Chin
        landmarks[42],  # Left eye left corner
        landmarks[39],  # Right eye right corne
        landmarks[54],  # Left Mouth corner
        landmarks[48]   # Right mouth corner

    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-165.0, 170.0, -135.0),  # Left eye left corner
        (165.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals

    #center = (size[1] / 2, size[0] / 2)
    #focal_length = size[1]                                    # 以摄像头的宽度（像素）代表焦距
    #focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)  # 焦距
    #focal_high = center[1] / np.tan(60 / 2 * np.pi / 180)  # 焦距
    #focal_length = 2.26 * 100
    focal_length = 890.6315
    focal_high = 888.5269
    camera_matrix = np.array(
        [[focal_length, 0, 0],
         [0, focal_high, 0],
         [267.2760, 278.8883, 1]], dtype="double"
    )

    #dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion   （距离系数/假设没有镜头失真）
    dist_coeffs = np.copy([[-0.5148],[2.1279],[-10.1076],[0],[0]])

    # 计算旋转矩阵rotation_vector和平移矩阵translation_vector
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    '''
    cv2.SOLVEPNP_ITERATIVE = 0

    SOLVEPNP_ITERATIVE的迭代方法是基于Levenberg - Marquardt优化。 在这种情况下，函数会找到一个使重新投影误差最小的位姿（pose），该位姿是观察到的投影imagePoints与使用projectPoints将objectPoints投影的点之间的平方距离的总和（参考）。
    Levenberg - Marquardt法（LM法）是一种非线性优化方法。LM算法用于解决非线性最小二乘问题，多用于曲线拟合等场合
    2、cv2.SOLVEPNP_EPNP = 1
    3、cv2.SOLVEPNP_P3P = 2
    4、cv2.SOLVEPNP_DLS = 3
    5、cv2.SOLVEPNP_UPNP = 4
    6、cv2.SOLVEPNP_AP3P = 5
    '''

    #axis = np.float32([[0, 0, 500],
    #                   [0, 500, 0],
    #                   [500, 0, 0]])

    #imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    #modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return (str(int(roll)), str(int(pitch)), str(int(yaw)))


def findCentroid(eyeLandmarks,im,thresh):  #计算质心
    # 创建眼睛蒙版
    EyePoints = eyeLandmarks
    eyeMask = np.zeros_like(im)
    cv2.fillConvexPoly(eyeMask, np.int32(EyePoints), (255, 255, 255))
    eyeMask = np.uint8(eyeMask)

    # 定位虹膜
    r = im[:, :, 2]
    _, binaryIm = cv2.threshold(r, thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    morph = cv2.dilate(binaryIm, kernel, 1)
    morph = cv2.merge((morph, morph, morph))
    morph = morph.astype(float) / 255
    eyeMask = eyeMask.astype(float) / 255
    iris = cv2.multiply(eyeMask, morph)

    # 寻找质心
    M = cv2.moments(iris[:,:,0])
    try:
        cX = round(M["m10"] / M["m00"])
        cY = round(M["m01"] / M["m00"])
    except:
        cX = 0
        cY = 0

    centroid = (cX, cY)
    return centroid

def ScreenTestPoint():
    txtX = open('Xtestdata.txt','w')
    txtY = open('Ytestdata.txt','w')

    faceDetector = dlib.get_frontal_face_detector()  # 人脸检测器
    landmarkDetector = dlib.shape_predictor('G:/python_test/model/shape_predictor_68_face_landmarks.dat')  # 人脸特征点检测器
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    p = ([960,540],[30,30],[1890,30],[1890,1050],[30,1050])
    pygame.init()  #初始化屏幕画图
    screen = pygame.display.set_mode([1920,1080])  #屏幕显示尺寸1920*1080

    #pygame.display.flip()  #显示
    for point in p:
        i = 0
        if point == p[0]:
            X = 0
        elif point == p[1]:
            X = -1
        elif point == p[2]:
            X = 1
        elif point == p[3]:
            X = 1
        elif point == p[4]:
            X = -1
        # 测试Y点
        if point == p[0]:
            Y = 0
        elif point == p[1]:
            Y = -1
        elif point == p[2]:
            Y = -1
        elif point == p[3]:
            Y = 1
        elif point == p[4]:
            Y = 1
        screen.fill([0,0,0])  #屏幕全黑
        pygame.draw.circle(screen, [255, 0, 0], point, 15, 0)
        pygame.display.flip()

        for i in range(100):

            ret, img = cap.read()  # 读取视频流的一帧
            rects = faceDetector(img, 1)  # 人脸检测
            if (len(rects) == 1) and (i>=10) and (i<=90):
                shape = landmarkDetector(img, rects[0])  # 检测特征点
                points = face_utils.shape_to_np(shape)  # convert the facial(x,y)-coordinates to a Nump array
                #获取内眼角坐标
                leftrightEye = points[39:42 + 1]  # 取内眼角对应的特征点坐标

                #获取质心
                leftEye = points[42:48]  # 取出左眼对应的特征点
                rightEye = points[36:42]  # 取出右眼对应的特征点
                leftIrisCentroid = findCentroid(leftEye,img,40)
                rightIrisCentroid = findCentroid(rightEye,img,40)
                if leftIrisCentroid[0]!=0 and leftIrisCentroid[1]!=0 and rightIrisCentroid[0]!=0 and rightIrisCentroid[0]!=0:
                    #计算头部姿态参数
                    #imgpts, modelpts, rotate_degree, nose = HPEMain.face_orientation(img, points)
                    rotate_degree = face_orientation(img, points)
                    #测试X点

                    txtX.write(str('[')+str(points[33][0])+','+str(points[33][1])+','+str(rotate_degree[0])+','+str(rotate_degree[1])+','+str(rotate_degree[2])+','+str(leftIrisCentroid[0])+','+str(rightIrisCentroid[0])+','+str(points[39][0])+','+str(points[42][0])+str(']')+','+str('[')+str(X)+str(']'))
                    txtX.write('\n')
                    txtY.write(str('[')+str(points[33][0]) + ',' + str(points[33][1]) + ',' + str(rotate_degree[0]) + ',' + str(rotate_degree[1]) + ',' + str(rotate_degree[2]) + ',' + str(leftIrisCentroid[1]) + ',' + str(rightIrisCentroid[1]) + ',' + str(points[39][1]) + ',' + str(points[42][1]) + str(']')+',' + str('[')+ str(Y)+str(']'))
                    txtY.write('\n')
                    #pygame.time.delay(100)
                    i += 1


        #pygame.time.delay(2000)
        
    txtX.close()
    txtY.close()
    pygame.QUIT
    
if __name__ == '__main__':
    ScreenTestPoint()