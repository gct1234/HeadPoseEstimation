# coding = utf-8
'''

date : 2022-4-24
desc:  头部姿态估计算法

'''

import cv2
from cv2 import SOLVEPNP_ITERATIVE
import dlib
import math
import numpy as np
from imutils import face_utils


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
    '''
    center = (size[1] / 2, size[0] / 2)
    #focal_length = size[1]                                    # 以摄像头的宽度（像素）代表焦距
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)  # 焦距
    focal_high = center[1] / np.tan(60 / 2 * np.pi / 180)  # 焦距
    #focal_length = 2.26 * 100
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_high, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion   （距离系数/假设没有镜头失真）
    '''
    focal_length = 890.6315
    focal_high = 888.5269
    camera_matrix = np.array(
        [[focal_length, 0, 0],
         [0, focal_high, 0],
         [267.2760, 278.8883, 1]], dtype="double"
    )

    # dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion   （距离系数/假设没有镜头失真）
    dist_coeffs = np.copy([[-0.5148], [2.1279], [-10.1076], [0], [0]])

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

    axis = np.float32([[0, 0, 500],
                       [0, 500, 0],
                       [500, 0, 0]])

    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[33])


def draw_pose():
    faceDetector = dlib.get_frontal_face_detector()  # 人脸检测器
    landmarkDetector = dlib.shape_predictor('G:/python_test/model/shape_predictor_68_face_landmarks.dat')  # 人脸特征点检测器
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while (1):
        ret, img = cap.read()  # 读取视频流的一帧

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转成灰度图像

        rects = faceDetector(gray, 0)  # 人脸检测


        for rect in rects:
            shape = landmarkDetector(img, rect)  # 检测特征点
            points = face_utils.shape_to_np(shape)  # convert the facial(x,y)-coordinates to a Nump array
            imgpts, modelpts, rotate_degree, nose = face_orientation(img, points)
            cv2.line(img, (nose[0],nose[1]), (round(imgpts[1][0][0]),round(imgpts[1][0][1])), (0, 255, 0), 3)  # GREEN
            cv2.line(img, (nose[0],nose[1]), (round(imgpts[0][0][0]),round(imgpts[0][0][1])), (255, 0, 0), 3)  # BLUE
            cv2.line(img, (nose[0],nose[1]), (round(imgpts[2][0][0]),round(imgpts[2][0][1])), (0, 0, 255), 3)  # RED
            #cv2.line(img, (points[33][0],points[33][1]), tuple(imgpts[2].ravel()), (0, 0, 255), 3)  # RED
            remapping = [2, 3, 0, 4, 5, 1]

            #for index in range(int(len(points) / 2)):
            #    random_color = tuple(np.random.random_integers(0, 255, size=3))

            #    cv2.circle(img, (points[index * 2], points[index * 2 + 1]), 5, random_color, -1)
            #    cv2.circle(img, tuple(modelpts[remapping[index]].ravel().astype(int)), 2, random_color, -1)

            #    cv2.putText(frame, rotate_degree[0]+' '+rotate_degree[1]+' '+rotate_degree[2], (10, 30),
            #                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
            #                thickness=2, lineType=2)

            for j in range(len(rotate_degree)):
                cv2.putText(img, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), thickness=2, lineType=2)

            # time.sleep(3)
            cv2.imshow("frame", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
   draw_pose()

