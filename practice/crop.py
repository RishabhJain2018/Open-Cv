import numpy as np 
import cv2
img=cv2.imread("pic1.png",0)
print img
print "Hello World"
cv2.imshow("google",img) #returns None
a = cv2.waitKey(0)
if a==27:
	cv2.destroyAllWindows()
	print cv2.imwrite('messigray1.png',img)  # returns True or False (boolean)
elif a == ord('s'):
	cv2.imwrite('messigray1.png',img)
	cv2.destroyAllWindows()