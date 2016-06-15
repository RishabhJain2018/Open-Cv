import cv2

img = cv2.imread("pic2.jpg")
crop_img = img[200:400, 200:400]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

