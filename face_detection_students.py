from urllib.request import urlopen
import numpy as np
import cv2
import sys
from PIL import Image

i = 0
frontal_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("Enter an id for the user (for example : 1, 2, 3) ")
print("Do not take 1 for example if there is student.1.0.jpg ")
id_student = input("Choose a number that is not part of the dataset folder : ")
ip = input("What is the ip of the camera ? ")
url = 'http://' + str(ip) + ':8080/shot.jpg'

while True:
    try:
        imgNp=np.array(bytearray(urlopen(url).read()),dtype=np.uint8)
    except Exception:
        print('IP is not valid please retry')
        break
    image_frame=cv2.imdecode(imgNp,-1)
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = frontal_face.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.imwrite("dataset/student." + str(id_student) + "." + str(i) + ".jpg", gray[y:y+h,x:x+w])
        i += 1
    cv2.imshow('face_student_detection', image_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if i > 100:
        break

