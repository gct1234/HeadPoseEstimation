# coding = utf-8
'''

date :  2022-4-25
desc:   在图片中画出头部姿态参数

'''

import cv2
import numpy as np

# Read Image
im = cv2.imread("G:/python_test/FaceLibs/face/27.jpg")
size = im.shape

#2D image points. If you change the image, you need to change vector
image_points = np.array([
                            (642,709),     # Nose tip  642	709
                            (679,877),     # Chin  679	877
                            (589,529),     # Left eye left corner
                            (715,531),     # Right eye right corne 715	531
                            (593,755),     # Left Mouth corner 593	755
                            (739,761)      # Right mouth corner 739	761
                        ], dtype="double")

# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner

                        ])


# Camera internals
focal_length = 890.6315
focal_high = 888.5269
camera_matrix = np.array(
        [[focal_length, 0, 0],
         [0, focal_high, 0],
         [267.2760, 278.8883, 1]], dtype="double"
    )

dist_coeffs = np.copy([[-0.5148],[2.1279],[-10.1076],[0],[0]])

'''
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
'''
print("Camera Matrix :\n {0}".format(camera_matrix))

#dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

print( "Rotation Vector:\n {0}".format(rotation_vector))
print( "Translation Vector:\n {0}".format(translation_vector))


# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose


(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

for p in image_points:
    cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


p1 = ( int(image_points[0][0]), int(image_points[0][1]))
p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

cv2.line(im, p1, p2, (255,0,0), 2)

# Display image
cv2.imshow("Output", im)
cv2.waitKey(0)
