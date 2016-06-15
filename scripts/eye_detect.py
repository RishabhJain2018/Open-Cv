from os import listdir
import os
import cv2
import sys

facecascPath = "/" + os.popen('pwd').read().strip('\n') + "/haarcascade_frontalface_default.xml"
eyecascPath="/" + os.popen('pwd').read().strip('\n') + "/haarcascade_eye.xml"
framesPath="/" + os.popen('pwd').read().strip('/scripts\n') + "/faceDetect/2.MOV/"

imageList = listdir(framesPath)

for img in imageList:

	faceCascade = cv2.CascadeClassifier(facecascPath)
	eyeCascade = cv2.CascadeClassifier(eyecascPath)

	image = cv2.imread(framesPath+str(img))
	if image == None:
		print str(img)
	else:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray, 1.3, 5)
		print "Found {0} eyes !".format(len(faces))

		for (x,y,w,h) in faces:

			roi_gray = gray[y:y+h, x:x+w]
			roi_color = image[y:y+h, x:x+w]

			eyes = eyeCascade.detectMultiScale(roi_gray)

			for (ex, ey, ew, eh) in eyes:
				cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(255,0,0),2)

				cv2.imwrite("/home/rishabh/Documents/Open-Cv/eyeDetect/"+str(img),image)
			
cv2.waitKey(0)