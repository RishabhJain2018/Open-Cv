import cv2
import os
from skimage.feature import local_binary_pattern
import cvutils
from scipy.stats import itemfreq
import numpy as np 

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image=cv2.imread('frame0.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor=1.3,
	minNeighbors=5,
	minSize=(30,30),
	flags=cv2.cv.CV_HAAR_SCALE_IMAGE,
	)

for(x,y,w,h) in faces:
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

	crop_image1 = image[y:y+h, x:x+w]

gray_img = cv2.cvtColor(crop_image1, cv2.COLOR_BGR2GRAY)
print gray_img.shape[1] 
windowsize_c = 10
windowsize_r = 10
radius=1
no_points = 8*radius
size=(90,90)
full_lbp = np.empty(size, dtype='float32')
print full_lbp.size
idx = 0
for r in range(0, gray_img.shape[0], windowsize_r):
	for c in range(0,gray_img.shape[1], windowsize_c):
		window=gray_img[r:r+windowsize_r, c:c+windowsize_c]
		lbp = local_binary_pattern(window,no_points,radius,method='nri_uniform')
		# print lbp

		# (hist,_) = np.histogram(lbp.ravel(),bins=np.arange(0,no_points+3),range=(0,no_points+2))
		np.seterr(divide='ignore', invalid='ignore')
		lbp = np.sqrt(np.divide(lbp*1.0,np.sum(lbp)))
		
		full_lbp = np.append(full_lbp,lbp)

		# full_lbp[idx:idx+100]=lbp
		# idx+=100

print full_lbp
