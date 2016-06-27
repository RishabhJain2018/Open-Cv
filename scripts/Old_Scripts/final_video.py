import cv2
import os
import numpy as np
from sklearn import svm
from skimage.feature import local_binary_pattern

# Cascade Files
cascadeFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascadeLEye = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
cascadeREye = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")
cascadeNose = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
cascadeMouth = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Save path
#path_retrieve = "S:/Users/Sahil Verma/PycharmProjects/iAugmentor/zzz/lfw2/"
#path_save = "S:/Users/Sahil Verma/PycharmProjects/iAugmentor/Facial Detection/"

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

    # Go through the list, picking up each image for facial detection
    #for file1 in listing1:
    #listing2 = os.listdir(path_retrieve + file1)

        #Go through each image within the folder
    for file2 in listing1:
        if file2.lower().endswith(('.png', '.jpg', '.jpeg')):

            LSLE = []
            RSLE = []
            LSRE = []
            RSRE = []
            Nose = []
            LSL = []
            RSL = []

            # Reading and displaying image
            image = cv2.imread(path_retrieve + file2)
            cv2.imshow("Original " + file2, image)
            cv2.waitKey()

            # Converting to gray, for reduction of noise
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # FACES
            # Reading the faces in each image and going through all of them
            faces = cascadeFace.detectMultiScale(gray, 1.5, 5)
            for (x, y, w, h) in faces:
                # Sliing te image te image
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = image[y:y + h, x:x + w]

                print("Found Face")

                # LEFT EYE
                # Reading the eyes in each image and going through all of them
                eyes = cascadeREye.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    # Co-ordinates for all eye points
                    LSLE_x = ex + int(ew/10)
                    LSLE_y = ey + int(eh/2)
                    RSLE_x = ex + int(0.9 * ew)
                    RSLE_y = ey + int(eh/2)

                    LSLE = [LSLE_x + x, LSLE_y + y]
                    RSLE = [RSLE_x + x, RSLE_y + y]

                    cv2.line(roi_color, (LSLE_x, LSLE_y),
                             (LSLE_x + 1, LSLE_y), (0, 0, 255), thickness)
                    cv2.line(roi_color, (RSLE_x, RSLE_y),
                             (RSLE_x + 1, RSLE_y), (0, 0, 255), thickness)

                    print("Found LEYE")
                    break

                # RIGHT EYE
                # Reading the right eye in each image and going through all of them
                eyes = cascadeLEye.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    # Co-ordinates for all eye points
                    LSRE_x = ex + int(ew / 10)
                    LSRE_y = ey + int(eh / 2)
                    RSRE_x = ex + int(0.9 * ew)
                    RSRE_y = ey + int(eh / 2)

                    if LSRE_x > RSLE_x:
                        LSRE = [LSRE_x + x, LSRE_y + y]
                        RSRE = [RSRE_x + x, RSRE_y + y]

                        cv2.line(roi_color, (LSRE_x, LSRE_y), (LSRE_x + 1, LSRE_y), (0, 0, 255), thickness)
                        cv2.line(roi_color, (RSRE_x, RSRE_y), (RSRE_x + 1, RSRE_y), (0, 0, 255), thickness)

                        print("Found REYE")
                        break

                # NOSE
                # Reading the nose ONCE in each image
                nose = cascadeNose.detectMultiScale(roi_gray)
                for (nx, ny, nw, nh) in nose:
                    # Co-ordinates for nose tip
                    Nose_x = nx + int(nw/2)
                    Nose_y = ny + int(nh/2)

                    if (Nose_y > RSLE_y) & (Nose_x > LSLE_x):
                        Nose = [Nose_x + x, Nose_y + y]

                        cv2.line(roi_color, (Nose_x, Nose_y), (Nose_x + 1, Nose_y + 1), (0, 255, 0), thickness)

                        print("Found Nose")
                        break

                # LIPS
                # Reading the lips in each image and going through all of them
                mouth = cascadeMouth.detectMultiScale(roi_gray)
                for (mx, my, mw, mh) in mouth:
                    # Co-ordinates for lip ends
                    LSL_x = mx + int(mw/10)
                    LSL_y = my + int(0.3*mh)
                    RSL_x = mx + int(0.9*mw)
                    RSL_y = my + int(0.3*mh)

                    # Mouth must be below nose (obviously)
                    if Nose_y < LSL_y:
                        LSL = [LSL_x + x, LSL_y + y]
                        RSL = [RSL_x + x, RSL_y + y]

                        cv2.line(roi_color, (LSL_x, LSL_y), (LSL_x + 1, LSL_y), (255, 0, 0), thickness)
                        cv2.line(roi_color, (RSL_x, RSL_y), (RSL_x + 1, RSL_y), (255, 0, 0), thickness)

                        print("Found Lips")
                        break

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

            # Print the points acquired
            print(file2 + ":", [LSLE, RSLE, LSRE, RSRE, Nose, LSL, RSL])

            # Show the points detected
            cv2.imshow(file2 + " w Facial Detection", image)
            cv2.waitKey(0)

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
            (h, w) = image.shape[:2]
            A = A[:, :2].T

            rotated = cv2.warpAffine(image, A, (w, h))
            cv2.imshow("Rotated " + file2, rotated)
            cv2.waitKey()

            # Resizing image dimensions
            r = height / image.shape[1]
            dim = (width, int(image.shape[0] * r))

            # Perform the resizing of the image and show it
            print "======"
            resized = cv2.resize(rotated, dim, interpolation = cv2.INTER_AREA)
            print resized.flags

#############################          Rishabh          ##################################


            faces = faceCascade.detectMultiScale(resized,1.3,5)
            for(x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

                crop_image1 = image[y:y+h, x:x+w]

            cv2.imshow("tight Crop", crop_image1)
            cv2.waitKey(0)
            gray_img = cv2.cvtColor(crop_image1, cv2.COLOR_BGR2GRAY)
            print "1"

            print "cropped image+++++++++", crop_image1.flags
            windowsize_r=10
            windowsize_c=10
            radius=1
            no_points = 8*radius
            size = (100,100)
            full_lbp = np.array(size, dtype='float32')
            for r in range(0, crop_image1.shape[0]-windowsize_r,windowsize_r):
                for c in range(0, crop_image1.shape[1]-windowsize_c, windowsize_c):
                    window= crop_image1[r:r+windowsize_r, c:c+windowsize_c]
                    print "size",window.shape
                    window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                    cv2.imshow("window", window)
                    cv2.waitKey(1)
                    print "---45ry38----",window.flags
                    lbp= local_binary_pattern(window, no_points, radius, method='nri_uniform')
                    print lbp
                    np.seterr(divide='ignore', invalid='ignore')
                    lbp = np.sqrt(np.divide(lbp*1.0, np.sum(lbp)))
                    lbp=lbp.ravel()
                    full_lbp=np.concatenate((full_lbp, lbp))
        t = np.array((100,100),dtype='float32')
        t = np.append(full_lbp, t)
    return t

# For setting path of different images classified on basis of smiling and not smiling.
for i in xrange(1,5):
    if i==1:#for all smiling images
        path_retrieve = "/home/rishabh/Images (zipped) Folder/smiling/"
        T1 = process(path_retrieve)
    elif i==2:#for all not smiling images
        path_retrieve = "/home/rishabh/Images (zipped) Folder/notsmiling/"
        T1 = process(path_retrieve)
    #for all testing images
    elif i==3:
        path_retrieve = "/home/rishabh/Images (zipped) Folder/testing1/"
        X1_test = process(path_retrieve)
    else:
        path_retrieve = "/home/rishabh/Images (zipped) Folder/testing2/"
        X2_test = process(path_retrieve)
    
sigma1 = 0.1
mean1 = 1
sigma2 = 0.1
mean2 = 2
# training 
x1 = sigma1 * T1 + mean1 
x2 = sigma2 * T2 + mean2

x1 = x1.reshape(50,2)
x2 = x2.reshape(50,2)

y1 = np.ones(len(x1))
y2 = np.ones(len(x2)) * -1

X_train = np.vstack((x1, x2))
y_train = np.hstack((y1, y2))

clf = svm.LinearSVC()
model = clf.fit(X_train, y_train)

# prediction
X1_test = X1_test.reshape(50,2)
y1_test = np.ones(len(X1_test))
X2_test = X2_test.reshape(50,2)
y2_test = np.ones(len(X2_test))

X_test = np.vstack((X1_test, X2_test))
y_test = np.hstack((y1_test, y2_test))
pred = clf.predict(X_test)
correct = np.sum(pred == y_test) *1.0/ len(y_test)
# total predictions 
print "%d out of %d predictions correct" % (np.sum(pred==y_test), len(pred))