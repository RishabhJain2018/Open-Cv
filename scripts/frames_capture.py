import cv2
import os

base_path = os.popen("pwd").read().replace("\n", "").replace("scripts", "")
video_path = base_path + "videos/2.MOV"

vidcap=cv2.VideoCapture(video_path)
success, image = vidcap.read()
count = 0
while success:
	success, image = vidcap.read()
	cv2.imwrite(base_path + "frames/2.MOV/frame%d.jpg" %  count, image)
	if cv2.waitKey(10) == 27:
		break

	count+=1

print "All Frames Captured"	

