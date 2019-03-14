from urllib.request import urlopen
import sys
import cv2
import numpy as np
import os
from tkinter import *

def stockip():
    global string_ip
    global ip
    string = string_ip.get()
    ip = string
    root.destroy()

root = Tk()
root.title('IP camera')
root.geometry("300x100")
label_ip = Label(root, text="What is the ip of the camera ?")
label_ip.pack()
string_ip = Entry(root)
string_ip.pack()
string_ip.focus_set()
b = Button(root,text='Confirm',command=stockip)
b.pack(side='bottom')
root.mainloop()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['Unknow', 'Hugo', 'Flavien', 'Clement', 'Jeremy']
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
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknow"
            confidence = "  {0}%".format(round(100 - confidence))
        cv2.putText(image_frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
    cv2.imshow('face_recognition_students',image_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
