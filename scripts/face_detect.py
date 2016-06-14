import sys
import cv2

imagePath = sys.argv[1]
print "4"
cascPath=sys.argv[2]
print "5"


faceCascade = cv2.CascadeClassifier(cascPath)
print "6"

image = cv2.imread(imagePath)
print "7"

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print "8"

faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor = 1.1,
	minNeighbors = 5,
	minSize = (30, 30),
	flags = cv2.cv.CV_HAAR_SCALE_IMAGE,
	)

print "9"

print "Found {0} faces !".format(len(faces))

print "10"

count = 0
for (x,y,w,h) in faces:
	cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0),2)
	# print img
	# imgg = cv2.imwrite("faces00%d.jpg" % count ,img)
	cv2.imwrite("faces00%d.jpg" % count ,image)
	# print "imgg : ", imgg


	count +=1

# print "11"

# cv2.imshow("faces Found", image)
# print "12"

print "13"


cv2.waitKey(0)


