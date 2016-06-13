import numpy as np 
import cv2
img=cv2.imread("images.png",0)
print img
cv2.imshow("google",img)
cv2.waitKey(0)
cv2.destroyWindow("google")