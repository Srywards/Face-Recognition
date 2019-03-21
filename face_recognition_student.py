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
		self.met = Metier()

	def face_detection(self):
		DetectionWindow(self)

	def train(self):
		self.met.training()

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
		self.menuButton = tk.Button(self.g, text='Main Menu', command=self.g.destroy)
		self.menuButton.grid(column=0,row=0)
		self.label_ip = tk.Label(self.g, text="What is the ip of the camera ?")
		self.label_ip.grid(columnspan=1)
		self.string_ip = tk.Entry(self.g)
		self.string_ip.grid(columnspan=1)
		self.string_ip.focus_set()
		self.confirmButton = tk.Button(self.g, text='Confirm', command=self.face_recognition)
		self.confirmButton.grid(column=0)
		self.met = Metier()

	def face_recognition(self):
		self.met.face_recognition(self.string_ip)

class DetectionWindow(UI):
	def __init__(self, mainUI):
		self.mainUI = mainUI
		self.g = tk.Toplevel()
		self.g.title('Face detection')
		self.g.geometry("350x180")
		self.menuButton = tk.Button(self.g, text='Main Menu', command=self.g.destroy)
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
		self.confirmButton = tk.Button(self.g, text='Confirm', command=self.face_detection)
		self.confirmButton.grid(column=0)
		self.met = Metier()

	def face_detection(self):
		self.met.face_detection(self.string_ip, self.string_id_student)

class Metier():
	def __init__(self):
		self.ip = ""
		self.id = 0
		self.id_student = ""
		self.recognizer = cv2.face.LBPHFaceRecognizer_create()
		if not os.path.isdir('dataset'):
			print("Dataset dir not found")
			UI.exit(self)
		if not os.path.isdir('trainer'):
		    print("Trainer dir not found, please create one")
		    UI.exit(self)
		if not os.path.isfile('haarcascade_frontalface_default.xml'):
			print("Haarcascade file not found")
			UI.exit(self)
		self.haarcascade_file = "haarcascade_frontalface_default.xml"
		self.faceCascade = cv2.CascadeClassifier(self.haarcascade_file)
		if not os.path.isfile('trainer/trainer.yml'):
			print("Trainer file not found")
			UI.exit(self)
		self.trainer_file = "trainer/trainer.yml"
		self.connect = sqlite3.connect(':memory:')
		self.db = self.connect.cursor()
		self.db.execute('''CREATE TABLE students
		(name, time)''')

	def ip_connect(self, url):
		try:
			imgNp = np.array(bytearray(urlopen(url).read()),dtype=np.uint8)
		except Exception:
			print('IP is not valid please retry')
			return
		image_frame = cv2.imdecode(imgNp, -1)
		gray = cv2.cvtColor(image_frame,cv2.COLOR_BGR2GRAY)
		faces = self.faceCascade.detectMultiScale(
			gray,
			scaleFactor = 1.2,
			minNeighbors = 5,
			minSize = (50, 50),
			)
		return image_frame, faces, gray

	def face_detection(self, string_ip, string_id_student):
		self.ip = string_ip.get()
		self.id_student = string_id_student.get()
		self.i = 0
		url = 'http://' + str(self.ip) + ':8080/shot.jpg'
		while True:
			image_frame, faces, gray = self.ip_connect(url)
			for (x,y,w,h) in faces:
				cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
				cv2.imwrite("dataset/student." + str(self.id_student) + "." + str(self.i) + ".jpg", gray[y:y+h,x:x+w])
				self.i += 1
			cv2.imshow('face_student_detection', image_frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyWindow('face_student_detection')
				break
			if self.i > 1:
				cv2.destroyWindow('face_student_detection')
				break

	def face_recognition(self, string_ip):
		self.ip = string_ip.get()
		self.recognizer.read(self.trainer_file)
		names = ['Unknow', 's1mple']
		url = 'http://' + str(self.ip) + ':8080/shot.jpg'
		while True:
			image_frame, faces, gray = ip_connect()
			for(x,y,w,h) in faces:
				cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
				self.id, confidence = self.recognizer.predict(gray[y:y+h,x:x+w])
				if (confidence < 100):
					self.id = names[self.id]
					self.db.execute("INSERT INTO students VALUES (?, ?);", (str(self.id), time.strftime("%A %d %B %Y %H:%M:%S")))
					confidence = "  {0}%".format(round(100 - confidence))
				else:
					self.id = "unknow"
					confidence = "  {0}%".format(round(100 - confidence))
				cv2.putText(image_frame, str(self.id), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
			cv2.imshow('face_recognition_students',image_frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				self.connect.commit()
				for row in self.db.execute('SELECT * FROM students'):
					print (row)
				self.connect.close()
				cv2.destroyWindow('face_recognition_students')
				break

	def training(self):
	    path = ('dataset')
	    print("Training in progress..")
	    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
	    faceSamples=[]
	    students = []
	    for imagePath in imagePaths:
		    PIL_img = Image.open(imagePath).convert('L')
		    img_numpy = np.array(PIL_img,'uint8')
		    id = int(os.path.split(imagePath)[-1].split(".")[1])
		    faces = self.faceCascade.detectMultiScale(img_numpy)
		    for (x,y,w,h) in faces:
			    faceSamples.append(img_numpy[y:y+h,x:x+w])
			    students.append(id)
	    self.recognizer.train(faceSamples, np.array(students))
	    self.recognizer.write(self.trainer_file)
	    print("Training done !")

if __name__ == "__main__":
	root = tk.Tk()
	mainMenu = UI(root)
	root.mainloop()
