import dlib
import cv2
import numpy as np
import urllib.request as urllib
import argparse
import json


# argument parser that expects an URL
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--imageurl", required=True, help="image URL to recognize some face")
args = vars(ap.parse_args())

face_detector = dlib.get_frontal_face_detector()
points_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# downloads the image from URL
image_response = urllib.urlopen(args["imageurl"])
image = np.asarray(bytearray(image_response.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)

# image = cv2.imread("fotos/treinamento/ronald.0.0.jpg")
faces = face_detector(image, 1)

if len(faces) > 1:
    print("More than 1 face detected")
    exit()
elif len(faces) == 0:
    print("No face detected")
    exit()

face_points = points_detector(image, faces[0])
face_description = face_recognizer.compute_face_descriptor(image, face_points)
face_description_list = [fd for fd in face_description]
face_description_json = json.dumps(face_description_list)
print(face_description_json)
