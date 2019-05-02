# Face-Recognition

Recognition of faces with IP Camera, OpenCV, Sqlite Database and Python !

## Overview

The main window where you can choose the step of the program you want :

![ui_2](https://user-images.githubusercontent.com/15232456/57099143-044a4980-6d1c-11e9-99b4-5b604a4291c3.png)

The program tree structure :

```bash
├── dataset
│   ├── student.*.*.jpg
├── face_recognition_students.py
├── haarcascade_frontalface_default.xml
├── trainer
    └── trainer.yml
```

**haarcascade_frontalface_default.xml** : default configuration file of OpenCV to detect frontal_faces

**face_recognition_students.py** : Python program with GUI with : Face Detection
                                                                  Training
                                                                  Face recognition

**Dataset** : The folder where your images are stored

**Trainer** : Folder to save the configuration file [trainer.yml] made from training_students.py

## Installation & Usage

Make sure to install the dependencies below and after execute :

```bash
python face_recognition_students.py
```

Then choose **Face Detection** step of the program.

This will create 100 images in dataset folder like this : 

![example_dataset](https://user-images.githubusercontent.com/15232456/54198618-4945c280-44c7-11e9-87c6-c65154c6deed.png)

When you have all your images, click on **Training** (It takes more or less time depending on the size of the dataset), a trainer.yml is create for the next and last step of the program.

![alarm (1)](https://user-images.githubusercontent.com/15232456/54199793-151fd100-44ca-11e9-9295-e85ec7363f20.png)

Before you click on Recognize, at **Line 148 in face_recognition_students.py** change the name that matches your id

For example i gave the id 1 for simple :

![id_simple](https://user-images.githubusercontent.com/15232456/54199692-d8ec7080-44c9-11e9-8617-823a4ef2142a.png)

# That's all folks !

<img src="https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif" width="400" height="400"/>

## Dependencies

* Python3
* OpenCV
* PIP
* Numpy
* Image
* sqlite3

## License

Face-recognition-student is licensed under the MIT license.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
