import cv2
import os
import numpy as np
from sklearn import svm
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
import sys
import dlib
import glob
from skimage import io
from sklearn.externals import joblib


predictor_path = "/home/iaugmentor/code/dlib/examples/build/shape_predictor_68_face_landmarks.dat"

TOTAL_IMAGES_SMILING = 249
TOTAL_IMAGES_NOT_SMILING = 249
SIZE_OF_IMAGE = 250*250

# Average points of eye corners, nose tip and lip corners from lfw dataset
LSLE_AVG = [93, 117]  # Left Side Left Eye Average Points
RSLE_AVG = [115, 114]  # Right Side Left Eye Average Points
LSRE_AVG = [138, 114]  # Left Side Right Eye Average Points
RSRE_AVG = [159, 115]  # Right Side Right Eye Average Points
Nose_AVG = [126, 137]  # Nose Average Points
LSL_AVG = [108, 160]  # Left Side Lips Average Points
RSL_AVG = [143, 160]  # Right Side Lips Average Points

# Co-ordinates for the 7 points' array
LSLE = []  # Left Side Left Eye Points
RSLE = []  # Right Side Left Eye Points
LSRE = []  # Left Side Right Eye Points
RSRE = []  # Right Side Left Ete Points
Nose = []  # Nose Points
LSL = []  # Left Side Lips Points
RSL = []  # Right Side Lips Points

# Co-ordinates required for the 7 points
LSLE_x = 0
LSLE_y = 0
RSLE_x = 0
RSLE_y = 0
LSRE_x = 0
LSRE_y = 0
RSRE_x = 0
RSRE_y = 0
Nose_x = 0
Nose_y = 0
LSL_x = 0
LSL_y = 0
RSL_x = 0
RSL_y = 0

# Values required
nx = 0
ny = 0
nw = 0
nh = 0
thickness = 3
width = 250
height = 250
def process(path_retrieve):
    # Retrieve the file list
    listing1 = os.listdir(path_retrieve)
    print(list(listing1))

    count =0
    for file2 in listing1:
        if file2.lower().endswith(('.png', '.jpg', '.jpeg')):
            print count,file2
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(predictor_path)

            img_dlib = io.imread(path_retrieve +  file2)
            img_cv = cv2.imread(path_retrieve + file2)


            dets = detector(img_dlib, 1)
            # print("Number of faces detected: {}".format(len(dets)))
            for k, d in enumerate(dets):

                shape = predictor(img_dlib, d)

                LSLE = (shape.part(36).x, shape.part(36).y)
                RSLE = (shape.part(39).x, shape.part(39).y)
                LSRE = (shape.part(42).x, shape.part(42).y)
                RSRE = (shape.part(45).x, shape.part(45).y)
                Nose = (shape.part(30).x, shape.part(30).y)
                LSL = (shape.part(60).x, shape.part(60).y)
                RSL = (shape.part(54).x, shape.part(54).y)

                cv2.line(img_cv, (LSLE[0], LSLE[1]),
                                     (LSLE[0] + 1, LSLE[1]), (0, 0, 255), thickness)
                cv2.line(img_cv, (RSLE[0], RSLE[1]),
                                     (RSLE[0] + 1, RSLE[1]), (0, 0, 255), thickness)
                cv2.line(img_cv, (LSRE[0], LSRE[1]),
                                     (LSRE[0] + 1, LSRE[1]), (0, 0, 255), thickness)
                cv2.line(img_cv, (RSRE[0], RSRE[1]),
                                     (RSRE[0] + 1, RSRE[1]), (0, 0, 255), thickness)
                cv2.line(img_cv, (Nose[0], Nose[1]),
                                     (Nose[0] + 1, Nose[1]), (0, 255, 0), thickness)
                cv2.line(img_cv, (LSL[0], LSL[1]),
                                     (LSL[0] + 1, LSL[1]), (255, 0, 0), thickness)
                cv2.line(img_cv, (RSL[0], RSL[1]),
                                     (RSL[0] + 1, RSL[1]), (255, 0, 0), thickness)

                # print("Showing Image")
                # cv2.imshow("image",img_cv)
                # cv2.waitKey()
                #win.add_overlay(shape)

                if LSLE == []:
                    LSLE = [LSLE_AVG[0], LSLE_AVG[1]]
                    RSLE = [RSLE_AVG[0], RSLE_AVG[1]]
                    print("No Left Eye")
                if LSRE == []:
                    LSRE = [LSRE_AVG[0], LSRE_AVG[1]]
                    RSRE = [RSRE_AVG[0], RSRE_AVG[1]]
                    print("No Right Eye")
                if Nose == []:
                    Nose = [Nose_AVG[0], Nose_AVG[1]]
                    print("No Nose")
                if LSL == []:
                    LSL = [LSL_AVG[0], LSL_AVG[1]]
                    RSL = [RSLE_AVG[0], RSLE_AVG[1]]
                    print("No Lips")

                # Points required
                pts1 = np.float32([LSLE, RSLE, LSRE, RSRE, Nose, LSL, RSL])
                pts2 = np.float32([LSLE_AVG, RSLE_AVG, LSRE_AVG, RSRE_AVG, Nose_AVG, LSL_AVG, RSL_AVG])

                # Pad the data with ones, so that our transformation can do translations too
                n = pts1.shape[0]
                pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
                unpad = lambda x: x[:, :-1]
                X = pad(pts1)
                Y = pad(pts2)

                # Solve the least squares problem X * A = Y, to find our transformation matrix A
                A, res, rank, s = np.linalg.lstsq(X, Y)
                transform = lambda x: unpad(np.dot(pad(x), A))

                # Grab the dimensions of the image, translate the image and show it
                (h, w) = img_cv.shape[:2]
                A = A[:, :2].T

                rotated = cv2.warpAffine(img_cv, A, (w, h))
                # cv2.imshow("Rotated ", rotated)
                # cv2.waitKey()

                # Resizing image dimensions
                r = height / img_cv.shape[1]
                dim = (width, int(img_cv.shape[0] * r))

                # Perform the resizing of the image and show it
                # print "======"
                resized = cv2.resize(rotated, dim, interpolation = cv2.INTER_AREA)
                # print resized.flags

                # cv2.destroyAllWindows()

#############################          Rishabh          ##################################

            gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            radius=1
            no_points = 8*radius

            lbp = local_binary_pattern(gray_img, no_points, radius, method='default')

            lbp = np.sqrt(np.divide(lbp*1.0, np.sum(lbp)))

            if count ==0:
                t=np.array(lbp)
                np.savetxt('initial_t.txt',t)
                count+=1
            else:
                t = np.concatenate((t,lbp))
                np.savetxt('final_t.txt',t) 
                count+=1
    return t



#*****************************************************
#****************start of main code ******************
#*****************************************************


# For setting path of different images classified on basis of smiling and not smiling.
for i in xrange(1,4):
    if i==1:#for all smiling images
        path_retrieve = "/home/iaugmentor/code/TrainingData/smiling/"
        T1 = np.array(process(path_retrieve))

    elif i==2:#for all not smiling images
        path_retrieve = "/home/iaugmentor/code/TrainingData/notsmiling/"
        T2 = np.array(process(path_retrieve))


# training 
x1 = T1
x2 = T2 

x1 = x1.reshape(TOTAL_IMAGES_SMILING, SIZE_OF_IMAGE)

x2 = x2.reshape(TOTAL_IMAGES_NOT_SMILING, SIZE_OF_IMAGE)

y1 = np.ones(TOTAL_IMAGES)
y2 = np.ones(TOTAL_IMAGES) * -1

X_train = np.vstack((x1, x2))
y_train = np.hstack((y1, y2))

clf = svm.LinearSVC()
model = clf.fit(X_train, y_train)

joblib.dump(clf, '/home/iaugmentor/code/model250/model.pkl')


