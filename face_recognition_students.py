from urllib.request import urlopen
import sys
import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['Unknow', 'Clement']
ip = input("What is the ip of the camera ? ")
url = 'http://' + str(ip) + ':8080/shot.jpg'

while True:
    try:
        imgNp=np.array(bytearray(urlopen(url).read()),dtype=np.uint8)
    except Exception:
        print('IP is not valid please retry')
        break
    image_frame=cv2.imdecode(imgNp,-1)
    gray = cv2.cvtColor(image_frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (50, 50),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 35):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknow"
            confidence = "  {0}%".format(round(100 - confidence))
        cv2.putText(image_frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(image_frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
    cv2.imshow('camera',image_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
