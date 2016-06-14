from os import listdir
from os.path import isfile, join
import os
import cv2
import sys

cascPath="/" + os.popen('pwd').read().strip('\n') + "/haarcascade_frontalface_default.xml"
framesPath="/" + os.popen('pwd').read().strip('/scripts\n') + "/frames/"

imageList = listdir(framesPath)

for img in imageList:

	faceCascade = cv2.CascadeClassifier(cascPath)
	image = cv2.imread(framesPath+str(img))
	if image == None:
		print str(img)
	else:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
		gray,
		stcaleFactor = 1.1,
		minNeighbors = 5,
		minSize = (30, 30),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE,
		)

		print "Found {0} faces !".format(len(faces))
		count = 0
		for (x,y,w,h) in faces:
			cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0),2)
			cv2.imwrite("/home/rishabh/Documents/Open-Cv/faceDetect/"+str(img),image)
			count +=1

cv2.waitKey(0)