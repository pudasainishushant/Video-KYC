import cv2
import numpy as np

from blink_liveness import check_blink


cap = cv2.VideoCapture(0)

cv2.namedWindow('BlinkDetector')
status = False
while True:

	retval, frame = cap.read()
	try:
		frame, status = check_blink(frame)
	except:
		frame=frame
		status = 1
	print(status)
	cv2.imshow('BlinkDetector', frame)

	key = cv2.waitKey(1)

	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()