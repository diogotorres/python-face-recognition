import dlib
import cv2
import glob
import os
import argparse
import json
from icrawler.builtin import GoogleImageCrawler
import logging


# number of images to download
images_to_download = 10
images_to_process = 5

# argument parser declaration
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--person", required=True, help="person name")
args = vars(ap.parse_args())

# logging configuration
logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

# load the models
face_detector = dlib.get_frontal_face_detector()
points_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# download directory
formatted_name = str(args["person"]).lower().replace(" ", "_")
training_dir = "images/training/{}".format(formatted_name)

logging.info("Starting crawl to {}".format(args["person"]))

# google crawler
crawler = GoogleImageCrawler(storage={"root_dir": training_dir})
crawler.crawl(keyword=args["person"], max_num=images_to_download)

logging.info("Finish crawling")

accepted_images = 0

for image_file in glob.glob(os.path.join(training_dir, "*")):

    if accepted_images == images_to_process:
        os.remove(image_file)
        continue

    print("Detecting faces on image {}".format(image_file))
    image = cv2.imread(image_file)
    faces = face_detector(image, 1)

    # check if there are multiple faces in the image
    if len(faces) > 1:
        print("More than 1 face detected on image {}, skipping...".format(image_file))
        os.remove(image_file)
    elif len(faces) == 0:
        print("No faces detected on image {}, skipping...".format(image_file))
        os.remove(image_file)
    else:
        print("One face detected successfully")
        accepted_images += 1
        print("Extracting facial points of the image {}".format(image_file))
        face_points = points_detector(image, faces[0])
        face_description = face_recognizer.compute_face_descriptor(image, face_points)
        face_description_list = [fd for fd in face_description]
        face_description_json = json.dumps(face_description_list)
        print("Face points:")
        print(face_description_json)





