from urllib.request import urlopen
import numpy as np
import cv2
import sys
from tkinter import *
from PIL import Image

def stocktext():
    global string_id_student
    global string_ip
    global id_student
    global ip
    string = string_id_student.get()
    string2 = string_ip.get()
    id_student = string
    ip = string2
    root.destroy()

root = Tk()
root.title('User Infos')
root.geometry("400x200")
label_id = Label(root, text="Enter an id for the user (for example : 1, 2, 3)\nDo not take 1 for example if there is student.1.0.jpg\nChoose a number that is not part of the dataset folder :")
label_id.pack()
string_id_student = Entry(root)
string_id_student.pack()
string_id_student.focus_set()
label_ip = Label(root, text="What is the ip of the camera ?")
label_ip.pack()
string_ip = Entry(root)
string_ip.pack()
b = Button(root,text='Confirm',command=stocktext)
b.pack(side='bottom')
root.mainloop()

print("This is your ID : ", id_student)
print("This is your IP : ", ip)
i = 0
frontal_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
        print("Detection complete, now execute training_students.py")
        break

