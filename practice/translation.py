import cv2
import numpy as np 

img = cv2.imread('pic1.jpg')
print img
cv2.imshow('image', img)
num_rows, num_cols = img.shape[:2]
print num_rows, num_cols
'''
translation_matrix = np.float32([[1,0,70], [0,1,110]])
img_translation = cv2.warpAffine(img, translation_matrix,(num_cols+70, num_rows+110))
cv2.imshow('Translation', img_translation)
cv2.waitKey(0)'''