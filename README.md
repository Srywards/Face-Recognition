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

```bash
python face_recognition_students.py
```

![recognize_simple](https://user-images.githubusercontent.com/15232456/54198160-151dd200-44c6-11e9-85f9-284bbf87375a.png)


![recognize_unknow](https://user-images.githubusercontent.com/15232456/54198190-236bee00-44c6-11e9-8d3f-e12b84f66f07.png)

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
