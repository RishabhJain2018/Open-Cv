
import cv2

img= cv2.imread('pic1.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
cv2.imshow('Y	channel',	yuv_img[:,	:,	0])
cv2.imshow('U	channel',	yuv_img[:,	:,	1])
cv2.imshow('V	channel',	yuv_img[:,	:,	2])
cv2.imshow('input image', gray_img)
cv2.imwrite('gray_img001.jpg', gray_img)
cv2.waitKey(0)


