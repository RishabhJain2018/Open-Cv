import sys
import cv2

imagePath = sys.argv[1]
cascPath=sys.argv[2]

faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor = 1.1,
	minNeighbors = 5,
	minSize = (30, 30),
	flags = cv2.cv.CV_HAAR_SCALE_IMAGE,
	)

print "Found {0} faces !".format(len(faces))

count = 0
for (x,y,w,h) in faces:
	cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0),2)
	cv2.imwrite("faces00%d.jpg" % count ,image)


	count +=1

cv2.waitKey(0)


