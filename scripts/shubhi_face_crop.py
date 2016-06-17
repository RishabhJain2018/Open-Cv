import cv2
from os import listdir
import os

basePath = os.popen("pwd").read().replace("\n", "").replace("scripts","")

cascPath = basePath + "scripts/haarcascade_frontalface_default.xml"
framesPath = basePath + "frames/2.mp4/"

imageList = listdir(framesPath)
count = 0
for img in imageList:
        faceCascade = cv2.CascadeClassifier(cascPath)
        image = cv2.imread(framesPath + str(img))

        if image==None:
                print image
        else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.5,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                )

                for (x, y, w, h) in faces:
                        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                count +=1
                sub_face = image[y:y+h, x:x+w]
                cv2.imwrite(basePath + "faceCrop/type1/shubhi_1.mp4/cropped00%d.jpg" %count, sub_face)

print 'face cropping done..!!'
cv2.waitKey(0)
