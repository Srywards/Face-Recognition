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
        self.master.geometry("155x140")
        self.label = tk.Label(master, text='Face recognition student')
        self.detectionButton = tk.Button(master, text='Face detection', command=self.face_detection)
        self.trainingButton = tk.Button(master, text='Training', command=self.train)
        self.recognizeButton = tk.Button(master, text='Recognize', command=self.recognize)
        self.exitButton = tk.Button(master, text='Exit', command=self.exit)
        self.label.grid(row =1, column=1)
        self.detectionButton.grid(row =2, column=1)
        self.trainingButton.grid(row =3, column=1)
        self.recognizeButton.grid(row =4, column=1)
        self.exitButton.grid(row =5, column=1)

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
		self.g.geometry("190x100")
		self.menuButton = tk.Button(self.g, text='Main Menu', command=self.exitMenu)
		self.menuButton.grid(column=0,row=0)
		self.label_ip = tk.Label(self.g, text="What is the ip of the camera ?")
		self.label_ip.grid(columnspan=1)
		self.string_ip = tk.Entry(self.g)
		self.string_ip.grid(columnspan=1)
		self.string_ip.focus_set()
		self.confirmButton = tk.Button(self.g, text='Confirm', command=self.confirm)
		self.confirmButton.grid(column=0)

	def exitMenu(self):
		self.g.destroy()

	def confirm(self):
		string = self.string_ip.get()
		self.ip = string
		id = 0
		if os.path.isfile('trainer/trainer.yml'):
			pass
		else:
			print("Trainer file not found")
			self.exitMenu()
			return
		recognizer = cv2.face.LBPHFaceRecognizer_create()
		recognizer.read('trainer/trainer.yml')
		if os.path.isfile('haarcascade_frontalface_default.xml'):
			pass
		else:
			print("Haarcascade file not found")
			self.exitMenu()
			return
		faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
		font = cv2.FONT_HERSHEY_SIMPLEX
		names = ['Unknow', 's1mple']
		url = 'http://' + str(self.ip) + ':8080/shot.jpg'
		connect = sqlite3.connect(':memory:')
		db = connect.cursor()
		db.execute('''CREATE TABLE students
		(name, time)''')
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
					db.execute("INSERT INTO students VALUES (?, ?);", (str(id), time.strftime("%A %d %B %Y %H:%M:%S")))
					confidence = "  {0}%".format(round(100 - confidence))
				else:
					id = "unknow"
					confidence = "  {0}%".format(round(100 - confidence))
				cv2.putText(image_frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
			cv2.imshow('face_recognition_students',image_frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				connect.commit()
				for row in db.execute('SELECT * FROM students'):
					print (row)
				connect.close()
				cv2.destroyWindow('face_recognition_students')
				self.exitMenu()
				break

class DetectionWindow(UI):
	def __init__(self, mainUI):
		self.mainUI = mainUI
		self.g = tk.Toplevel()
		self.g.title('Face detection')
		self.g.geometry("350x180")
		self.menuButton = tk.Button(self.g, text='Main Menu', command=self.exitMenu)
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
		self.confirmButton = tk.Button(self.g, text='Confirm', command=self.confirm)
		self.confirmButton.grid(column=0)

	def exitMenu(self):
		self.g.destroy()

	def confirm(self):
		string = self.string_id_student.get()
		string2 = self.string_ip.get()
		self.id_student = string
		self.ip = string2
		i = 0
		if os.path.isfile('haarcascade_frontalface_default.xml'):
			pass
		else:
			print("Haarcascade file not found")
			self.exitMenu()
			return
		recognizer = cv2.face.LBPHFaceRecognizer_create()
		frontal_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		url = 'http://' + str(self.ip) + ':8080/shot.jpg'
		while True:
			try:
				imgNp = np.array(bytearray(urlopen(url).read()), dtype=np.uint8)
			except Exception:
				print('IP is not valid please retry')
				break
			image_frame = cv2.imdecode(imgNp, -1)
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
				cv2.destroyWindow('face_student_detection')
				break
			if i > 1:
				cv2.destroyWindow('face_student_detection')
				self.exitMenu()
				break

class TrainingWindow(UI):
    def __init__(self, mainUI):
	    if os.path.isfile('haarcascade_frontalface_default.xml'):
		    pass
	    else:
		    print("Haarcascade file not found")
		    return
	    recognizer = cv2.face.LBPHFaceRecognizer_create()
	    frontal_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
	    if os.path.isdir('dataset'):
		    pass
	    else:
		    print("Dataset dir not found")
		    return
	    if os.path.isdir('trainer'):
		    pass
	    else:
		    print("Trainer dir not found, please create one")
		    return
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
