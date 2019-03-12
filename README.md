# Face-recognition-student

Recognition of students with IP Camera, OpenCV and Python !

## Overview

```bash
├── dataset
│   ├── student.*.*.jpg
├── face_detection_students.py
├── face_recognition_students.py
├── haarcascade_frontalface_default.xml
├── trainer
│   └── trainer.yml
└── training_students.py
```

**haarcascade_frontalface_default.xml** : default configuration file of OpenCV to detect frontal_faces

**face_detection_students.py** : Python program to detect the face of the student and store 100 images in the dataset folder

**training_students.py** : Python program to train the images and create a configuration file

**face_recognition_students.py** : Python program to detect students using the trainer.yml file

**Dataset** : Folder to save your images

**Trainer** : Folder to save the configuration file made from training_students.py

## Installation & Usage

Make sure to install the dependencies below and after execute :

```bash
python face_detection_students.py
```

![github_example](https://user-images.githubusercontent.com/15232456/54197679-d9ced380-44c4-11e9-9d45-3be4187a7dd0.png)

This will create 100 images in dataset folder like this : 

![example_dataset](https://user-images.githubusercontent.com/15232456/54198618-4945c280-44c7-11e9-87c6-c65154c6deed.png)

```bash
python training_students.py
```

![github_training](https://user-images.githubusercontent.com/15232456/54197821-3fbb5b00-44c5-11e9-8b7b-1b3378495f65.png)

![alarm (1)](https://user-images.githubusercontent.com/15232456/54199793-151fd100-44ca-11e9-9295-e85ec7363f20.png)

Before your run face_recognition_students.py, at **Line 13 in face_recognition_students.py** change the name that matches your id

For example i gave the id 1 for simple :

![id_simple](https://user-images.githubusercontent.com/15232456/54199692-d8ec7080-44c9-11e9-8617-823a4ef2142a.png)

```bash
python face_recognition_students.py
```

![recognition_simple](https://user-images.githubusercontent.com/15232456/54199234-b60d8c80-44c8-11e9-94b8-fa88676fa4b3.png)


![recognition_unknow](https://user-images.githubusercontent.com/15232456/54199291-d9d0d280-44c8-11e9-8e43-a88f8f1278c4.png)

# That's all folks !

<img src="https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif" width="400" height="400"/>

## Dependencies

* Python3
* OpenCV
* PIP
* Numpy
* Image

## License

Face-recognition-student is licensed under the MIT license.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
