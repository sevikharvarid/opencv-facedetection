import cv2
import numpy as np
from PIL import *

videoCam = cv2.VideoCapture(0)

face = cv2.CascadeClassifier('face-detect.xml')
eye = cv2.CascadeClassifier('eye-detect.xml')

rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("./recognizer/training3.yml")
id=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
while True:
    cond, frame = videoCam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    muka = face.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in muka:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 5)
	roi_warna = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        mata = eye.detectMultiScale(roi_gray)
        for (mx,my,mw,mh)in mata:
            cv2.rectangle(roi_warna, (mx,my), (mx+mw, my+mh), (255,255,0), 2)
	    id,conf=rec.predict(gray[y:y+h, x:x+w])
	    if(id==1):
		id="Donny"
	    elif(id==2):
		id="Vikhar"
	    elif(id==3):
		id="Agung"
	    else:
		id="Ngomul"
	    cv2.putText(frame, str(id), (x,y+h), font, 2, (0,8,255), 2, cv2.LINE_AA) 
	#cv2.putText(frame,id,(x,y+h),font,255)
        cv2.imshow('Face dan Eye detection', frame)

    if (cv2.waitKey(1)==ord('q')):
    	break

videoCam.release()
cv2.destroyAllWindows()
