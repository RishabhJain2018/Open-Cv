#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import os
import dlib
import glob
import cv2
import numpy as np
from skimage import io

# Average points of eye corners, nose tip and lip corners from lfw dataset
LSLE_AVG = [93, 117]  # Left Side Left Eye Average Points
RSLE_AVG = [115, 114]  # Right Side Left Eye Average Points
LSRE_AVG = [138, 114]  # Left Side Right Eye Average Points
RSRE_AVG = [159, 115]  # Right Side Right Eye Average Points
Nose_AVG = [126, 137]  # Nose Average Points
LSL_AVG = [108, 160]  # Left Side Lips Average Points
RSL_AVG = [143, 160]  # Right Side Lips Average Points

LSLE = []
RSLE = []
LSRE = []
RSRE = []
Nose = []
LSL = []
RSL = []
thickness = 2
height = 250
width = 250

predictor_path = "/home/iaugmentor/code/dlib/examples/build/shape_predictor_68_face_landmarks.dat"
faces_folder_path = "/home/iaugmentor/Images/notsmiling/"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img_dlib = io.imread(f)
    img_cv = cv2.imread(f)

    #win.clear_overlay()
    #win.set_image(img)img

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img_dlib, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        # print("Detection {}: Left:{} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
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

        print("Showing Image")
        cv2.imshow("image" + f,img_cv)
        cv2.waitKey()
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
        cv2.imshow("Rotated " + f, rotated)
        cv2.waitKey()

        # Resizing image dimensions
        r = height / img_cv.shape[1]
        dim = (width, int(img_cv.shape[0] * r))

        # Perform the resizing of the image and show it
        print "======"
        resized = cv2.resize(rotated, dim, interpolation = cv2.INTER_AREA)
        print resized.flags

        cv2.destroyAllWindows()

    # win.add_overlay(dets)
    #dlib.hit_enter_to_continue()



