from os import listdir
import os
import cv2

facecascPath=os.popen('pwd').read().replace("\n", "") + "/haarcascade_frontalface_default.xml"
framesPath=os.popen('pwd').read().replace("\n", "").replace("scripts", "") + "practice/"

imageList = listdir(framesPath)
padding = 15
count =0

for img in imageList:

	faceCascade = cv2.CascadeClassifier(facecascPath)
	image = cv2.imread(framesPath+str(img))
	if image == None:
		print str(img)
	else:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor = 1.5,
		minNeighbors = 5,
		minSize = (30, 30),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE,
		)

		print faces
		print "Hello World"
		for (x,y,w,h) in faces:
			print faces
			# print "face is :" 
			# print faces, x,y,w,h
			cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0),2)
			#type 1 cropping

		
			print y
			print y+h
			print x
			print x+w
			# crop_image1 = image[x:x+w, y:y+h]
			crop_image1 = image[y:y+h, x:x+w]

		cv2.imwrite("/home/rishabh/Documents/Open-Cv/practice/crop00%d.jpg" %count, crop_image1)
			#type 2 cropping
			# crop_image2 = image[y-padding:y+h+padding, x-padding:x+w+padding]
			# cv2.imwrite("/home/rishabh/Documents/Open-Cv/faceCrop/type2/2.mp4/"+ str(img), crop_image2)

			
print "Face Cropped...!!!"			
cv2.waitKey(0)
