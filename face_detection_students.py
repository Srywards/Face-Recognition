from urllib.request import urlopen
import numpy as np
import cv2
import sys
import os
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

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    students = []
    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = frontal_face.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            students.append(id)

    return faceSamples,students

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
recognizer = cv2.face.LBPHFaceRecognizer_create()
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
        cv2.destroyWindow('face_student_detection')
        break

if i > 1:
    print("Please wait, training in progress...\nIt may take longer depending on the size of the dataset")
    faces,students = getImagesAndLabels('dataset')
    recognizer.train(faces, np.array(students))
    recognizer.write('trainer/trainer.yml')
    print("You can now recognize the faces of the students, they will be stored in a database")
