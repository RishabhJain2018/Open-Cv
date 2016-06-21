import cv2
import os
import numpy as np

#Cascade Files
cascadeFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascadeLEye = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
cascadeREye = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")
cascadeNose = cv2.CascadeClassifier("nose2.xml")
cascadeMouth = cv2.CascadeClassifier("Mouth2.xml")

#Save path
path_retrieve = "S:/Users/Sahil Verma/PycharmProjects/iAugmentor/zzz/lfw2/"
path_save = "S:/Users/Sahil Verma/PycharmProjects/iAugmentor/Facial Detection/"
#text_name = "test.txt"

#Avergae points of eye corners, nose tip and lip corners from lfw dataset
LSLE_AVG = [93, 117]
RSLE_AVG = [115, 114]
LSRE_AVG = [138, 114]
RSRE_AVG = [159, 115]
Nose_AVG = [126, 137]
LSL_AVG = [108, 160]
RSL_AVG = [143, 160]

LSLE = []
RSLE = []
LSRE = []
RSRE = []
Nose = []
LSL = []
RSL = []

#Values required
nx = 0
ny = 0
nw = 0
nh = 0
thickness = 3
width = 250
height = 250

#Retrieve the file list
listing1 = os.listdir(path_retrieve)
print (list(listing1))

#with open("test.txt", "w") as myfile:
#    myfile.write("Facial Points")

#Go through the list, picking up each image for facial detection
for file1 in listing1:
    listing2 = os.listdir(path_retrieve + file1)

    for file2 in listing2:
        if file2.lower().endswith(('.png', '.jpg', '.jpeg')):

            # Reading and displaying image
            image = cv2.imread(path_retrieve + file1 + "/"+ file2)
            cv2.imshow("Original " + file2, image)
            cv2.waitKey()

            #with open(text_name, "a") as myfile:
            #    myfile.write("\nIMAGE: " + str(file2))
            #print("IMAGE: " + str(file2))

            # Converting to gray, for reduction of noise
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


            #FACES
            #Reading the faces in each image and going through all of them
            faces = cascadeFace.detectMultiScale(gray, 1.5, 5)
            for (x, y, w, h) in faces:
                #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = image[y:y + h, x:x + w]

                #LEFT EYE
                #Reading the eyes in each image and going through all of them
                eyes = cascadeREye.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    #Co-ordinates for all eye points
                    left_side_of_eye_x = ex + int(ew/10)
                    left_side_of_eye_y = ey + int(eh/2)
                    right_side_of_eye_x = ex + int(0.9 * ew)
                    right_side_of_eye_y = ey + int(eh/2)

                    LSLE = [left_side_of_eye_x + x, left_side_of_eye_y + y]
                    RSLE = [right_side_of_eye_x + x, right_side_of_eye_y + y]

                    cv2.line(roi_color, (left_side_of_eye_x, left_side_of_eye_y),
                             (left_side_of_eye_x + 1, left_side_of_eye_y), (0, 0, 255), thickness)
                    cv2.line(roi_color, (right_side_of_eye_x, right_side_of_eye_y),
                             (right_side_of_eye_x + 1, right_side_of_eye_y), (0, 0, 255), thickness)

                    #with open(text_name, "a") as myfile:
                    #    myfile.write(" LSLE: (" + str(left_side_of_eye_x) + "," + str(left_side_of_eye_y) + ") ")
                    #    myfile.write(" RSLE: (" + str(right_side_of_eye_x) + "," + str(right_side_of_eye_y) + ") ")

                    #print("LSLE: (" + str(left_side_of_eye_x) + "," + str(left_side_of_eye_y) + ") ")
                    #print("RSLE: (" + str(right_side_of_eye_x) + "," + str(right_side_of_eye_y) + ") ")


                #RIGHT EYE
                #Reading the right eye in each image and going through all of them
                eyes = cascadeLEye.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    #Co-ordinates for all eye points
                    left_side_of_eye_x = ex + int(ew / 10)
                    left_side_of_eye_y = ey + int(eh / 2)
                    right_side_of_eye_x = ex + int(0.9 * ew)
                    right_side_of_eye_y = ey + int(eh / 2)

                    LSRE = [left_side_of_eye_x + x, left_side_of_eye_y + y]
                    RSRE = [right_side_of_eye_x + x, right_side_of_eye_y + y]

                    cv2.line(roi_color, (left_side_of_eye_x, left_side_of_eye_y),
                             (left_side_of_eye_x + 1, left_side_of_eye_y), (0, 0, 255), thickness)
                    cv2.line(roi_color, (right_side_of_eye_x, right_side_of_eye_y),
                             (right_side_of_eye_x + 1, right_side_of_eye_y), (0, 0, 255), thickness)

                    #with open(text_name, "a") as myfile:
                    #   myfile.write(" LSRE: (" + str(left_side_of_eye_x) + "," + str(left_side_of_eye_y) + ") ")
                    #   myfile.write(" RSRE: (" + str(right_side_of_eye_x) + "," + str(right_side_of_eye_y) + ") ")

                    #print("LSRE: (" + str(left_side_of_eye_x) + "," + str(left_side_of_eye_y) + ") ")
                    #print("RSRE: (" + str(right_side_of_eye_x) + "," + str(right_side_of_eye_y) + ") ")


                #NOSE
                #Reading the nose ONCE in each image
                nose = cascadeNose.detectMultiScale(roi_gray)
                for (nx, ny, nw, nh) in nose:
                    #Co-ordinates for nose tip
                    nose_tip_x = nx + int(nw/2)
                    nose_tip_y = ny + int(nh/2)

                    cv2.line(roi_color, (nose_tip_x, nose_tip_y),
                             (nose_tip_x + 1, nose_tip_y + 1), (0, 255, 0), thickness)

                    Nose = [nose_tip_x + x, nose_tip_y + y]

                    #with open(text_name, "a") as myfile:
                    #    myfile.write("Nose: (" + str(nose_tip_x) + "," + str(nose_tip_y) + ") ")

                    #print("Nose: (" + str(nose_tip_x) + "," + str(nose_tip_y) + ") ")
                    break


                #LIPS
                #Reading the lips in each image and going through all of them
                mouth = cascadeMouth.detectMultiScale(roi_gray)
                for (mx, my, mw, mh) in mouth:
                    # Co-ordinates for lip ends
                    left_side_of_lips_x = mx + int(mw/10)
                    left_side_of_lips_y = my + int(0.3*mh)
                    right_side_of_lips_x = mx + int(0.9*mw)
                    right_side_of_lips_y = my + int(0.3*mh)

                    #Mouth must be below nose (obviously)
                    if (ny + int(0.6*nh) < my):
                        cv2.line(roi_color, (left_side_of_lips_x, left_side_of_lips_y),
                                 (left_side_of_lips_x + 1, left_side_of_lips_y), (255, 0, 0), thickness)
                        cv2.line(roi_color, (right_side_of_lips_x, right_side_of_lips_y),
                                 (right_side_of_lips_x + 1, right_side_of_lips_y),(255, 0, 0), thickness)

                    LSL = [left_side_of_lips_x + x, left_side_of_lips_y + y]
                    RSL = [right_side_of_lips_x + x, right_side_of_lips_y + y]

                    #with open(text_name, "a") as myfile:
                    #    myfile.write(" LSL: (" + str(left_side_of_lips_x) + "," + str(left_side_of_lips_y) + ") ")
                    #    myfile.write(" RSL: (" + str(right_side_of_lips_x) + "," + str(right_side_of_lips_y) + ") ")

                    #print("LSL: (" + str(left_side_of_lips_x) + "," + str(left_side_of_lips_y) + ") ")
                    #print("RSL: (" + str(right_side_of_lips_x) + "," + str(right_side_of_lips_y) + ") ")
                    break

            #Points required
            pts1 = np.float32([LSLE, RSLE, LSRE, RSRE, Nose, LSL, RSL])
            pts2 = np.float32([LSLE_AVG, RSLE_AVG, LSRE_AVG, RSRE_AVG, Nose_AVG, LSL_AVG, RSL_AVG])

            print (file2 + ":", [faces, LSLE, RSLE, LSRE, RSRE, Nose, LSL, RSL])

            #Pad the data with ones, so that our transformation can do translations too
            n = pts1.shape[0]
            pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
            unpad = lambda x: x[:, :-1]
            X = pad(pts1)
            Y = pad(pts2)

            #Solve the least squares problem X * A = Y, to find our transformation matrix A
            A, res, rank, s = np.linalg.lstsq(X, Y)
            transform = lambda x: unpad(np.dot(pad(x), A))

            #cv2.rectangle(image, (93, 117), (94, 118), (255, 0, 0), 2)
            #cv2.rectangle(image, (159, 115), (160, 116), (255, 0, 0), 2)
            #cv2.rectangle(image, (126, 137), (127, 138), (255, 0, 0), 2)

            cv2.imshow(file2 + " w Facial Detection", image)
            cv2.waitKey(0)

            #Grab the dimensions of the image and calculate the center of the image
            (h, w) = image.shape[:2]
            center = (w / 2, h / 2)

            #Rotating and translating the image
            A = A[:,:2].T
            rotated = cv2.warpAffine(image, A, (w, h))

            #cv2.rectangle(rotated, (93, 117), (94, 118), (0, 255, 0), 2)
            #cv2.rectangle(rotated, (159, 115), (160, 116), (0, 255, 0), 2)
            #cv2.rectangle(rotated, (126, 137), (127, 138), (0, 255, 0), 2)

            cv2.imshow("Rotated " + file2, rotated)
            cv2.waitKey()

            #Resizing image
            r = height / image.shape[1]
            dim = (width, int(image.shape[0] * r))

            #Perform the actual resizing of the image and show it
            resized = cv2.resize(rotated, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow("Resized " + file2, resized)
            cv2.waitKey()
            cv2.destroyAllWindows()

            #Showing the original for comparison
            #image = cv2.imread("Jan_Pronk_0001.jpg")

            #cv2.rectangle(image, (93, 117), (94, 118), (255, 0, 0), 2)
            #cv2.rectangle(image, (159, 115), (160, 116), (255, 0, 0), 2)
            #cv2.rectangle(image, (126, 137), (127, 138), (255, 0, 0), 2)

            #cv2.imshow("Original-2", image)
            #cv2.waitKey()

