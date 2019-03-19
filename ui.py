from urllib.request import urlopen
import numpy as np
import cv2
import sys
import os
import time
import sqlite3
import tkinter as tk
from PIL import Image

class UI():
    def __init__(self, master):
        self.master = master
        self.master.title("Face recognition student")
        self.master.geometry("150x150")
        self.label = tk.Label(master, text= 'Face recognition student')
        self.detectionButton = tk.Button(master, text= 'Face detection', command= self.face_detection)
        self.trainingButton = tk.Button(master, text= 'Training', command= self.train)
        self.recognizeButton = tk.Button(master, text= 'Recognize', command= self.recognize)
        self.exitButton = tk.Button(master, text= 'Exit', command= self.exit)
        self.label.grid(columnspan=2)
        self.detectionButton.grid(column=0)
        self.trainingButton.grid(column=0)
        self.recognizeButton.grid(column=0)
        self.exitButton.grid(column=0)

    def face_detection(self):
        DetectionWindow(self)

    def train(self):
        TrainingWindow(self)

    def recognize(self):
        RecognizeWindow(self)

    def exit(self):
        root.destroy()

class RecognizeWindow(UI):
	def __init__(self, mainUI):
		self.mainUI = mainUI
		self.g = tk.Toplevel()
		self.g.title('Face recognition')
		self.g.geometry("400x200")
		self.menuButton = tk.Button(self.g, text= 'Main Menu', command= self.exitMenu)
		self.menuButton.grid(column=0,row=0)
		self.label_ip = tk.Label(self.g, text="What is the ip of the camera ?")
		self.label_ip.grid(columnspan=1)
		self.string_ip = tk.Entry(self.g)
		self.string_ip.grid(columnspan=1)
		self.string_ip.focus_set()
		self.confirmButton = tk.Button(self.g, text= 'Confirm', command= self.confirm)
		self.confirmButton.grid(column=0)

	def exitMenu(self):
		self.g.destroy()

class DetectionWindow(UI):
	def __init__(self, mainUI):
		self.mainUI = mainUI
		self.g = tk.Toplevel()
		self.g.title('Face detection')
		self.g.geometry("400x200")
		self.menuButton = tk.Button(self.g, text= 'Main Menu', command= self.exitMenu)
		self.menuButton.grid(column=0,row=0)
		self.label_id = tk.Label(self.g, text="Enter an id for the user (for example : 1, 2, 3)\nDo not take 1 for example if there is student.1.0.jpg\nChoose a number that is not part of the dataset folder :")
		self.label_id.grid(columnspan=1)
		self.string_id_student = tk.Entry(self.g)
		self.string_id_student.grid(columnspan=1)
		self.string_id_student.focus_set()
		self.label_ip = tk.Label(self.g, text="What is the ip of the camera ?")
		self.label_ip.grid(columnspan=1)
		self.string_ip = tk.Entry(self.g)
		self.string_ip.grid(columnspan=1)
		self.confirmButton = tk.Button(self.g, text= 'Confirm', command= self.confirm)
		self.confirmButton.grid(column=0)

	def exitMenu(self):
		self.g.destroy()

	def confirm(self):
		string = self.string_id_student.get()
		string2 = self.string_ip.get()
		self.id_student = string
		self.ip = string2
		i = 0
		recognizer = cv2.face.LBPHFaceRecognizer_create()
		frontal_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		url = 'http://' + str(self.ip) + ':8080/shot.jpg'
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
				cv2.imwrite("dataset/student." + str(self.id_student) + "." + str(i) + ".jpg", gray[y:y+h,x:x+w])
				i += 1
			cv2.imshow('face_student_detection', image_frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			if i > 1:
				cv2.destroyWindow('face_student_detection')
				self.exitMenu()
				break

class TrainingWindow(UI):
    def __init__(self, mainUI):
	    recognizer = cv2.face.LBPHFaceRecognizer_create()
	    frontal_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
	    path = ('dataset')
	    print("Training in progress..")
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
	    recognizer.train(faceSamples, np.array(students))
	    recognizer.write('trainer/trainer.yml')
	    print("Training done !")

if __name__ == "__main__":
	root = tk.Tk()
	mainMenu = UI(root)
	root.mainloop()