import cv2
import os

base_path = os.popen("pwd").read().replace("\n", "").replace("scripts", "")

video_path = base_path + "videos/1.mp4"
facecascPath=os.popen('pwd').read().replace("\n", "") + "/haarcascade_frontalface_default.xml"
eyecascPath=os.popen('pwd').read().replace("\n", "") + "/haarcascade_eye.xml"

vidcap=cv2.VideoCapture(video_path)
success, image = vidcap.read()
frames = 0
cropframes1=0
cropframes2=0

while success:
	success, image = vidcap.read()
	frames+=1

	faceCascade = cv2.CascadeClassifier(facecascPath)
	eyeCascade = cv2.CascadeClassifier(eyecascPath)

	if image == None:
		pass
	else:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
			gray, 
			scaleFactor= 1.1,
			minNeighbors= 5,
			minSize = (30,30),
			flags = cv2.cv.CV_HAAR_SCALE_IMAGE,
			)

		padding = 15

		for (x,y,w,h) in faces:

			# for face
			cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0),2)

			# type 1 cropping
			crop_image1 = image[y:y+h, x:x+w]
			cropframes1+=1

			#type 2 cropping
			crop_image2 = image[y-padding:y+h+padding, x-padding:x+w+padding]
			cropframes2+=1

			roi_color = image[y:y+h, x:x+w]
			roi_gray = gray[y:y+h, x:x+w]

			# Commands for writing it to a folder.
			# cv2.imwrite("/home/rishabh/Documents/Open-Cv/final/1.mp4/crop1/frames00%d.jpg" %cropframes1,crop_image1)
			# cv2.imwrite("/home/rishabh/Documents/Open-Cv/final/1.mp4/crop2/frames00%d.jpg" %cropframes2,crop_image2)

			#for eyes
			eyes = eyeCascade.detectMultiScale(roi_gray)
			for (ex, ey, ew, eh) in eyes:
				cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh),(255,0,0),2)

				# print "found {0} eyes..!!".format(len(eyes))
		
print "Efficiency1 : {} ".format(frames-cropframes1)
print "Efficiency2 : {}".format(frames-cropframes2)

print "All done..!!"
cv2.waitKey(0)
