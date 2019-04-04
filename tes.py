import cv2

videoCam = cv2.VideoCapture(0)

face = cv2.CascadeClassifier('face-detect.xml')
eye = cv2.CascadeClassifier('eye-detect.xml')

id=raw_input("Masukkan ID : ")
sampleNum = 0
while True:
    cond, frame = videoCam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    muka = face.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in muka:
	sampleNum = sampleNum+1
	cv2.imwrite("dataWajah/User."+str(id)+"."+str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 5)
	cv2.waitKey(100)        
	roi_warna = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        mata = eye.detectMultiScale(roi_gray)
        for (mx,my,mw,mh)in mata:
            cv2.rectangle(roi_warna, (mx,my), (mx+mw, my+mh), (255,255,0), 2)

    cv2.imshow('Face dan Eye detection', frame)

    #k = cv2.waitKey(1) & 0xff
    cv2.waitKey(1)
    if(sampleNum>100):
        break

videoCam.release()
cv2.destroyAllWindows()
