import os
import glob
import cv2
import face_recognition
import numpy as np


class FaceSave(object):
    def __init__(self, save_image_dir):
        self.save_image_dir = save_image_dir
        super(FaceSave, self).__init__()

    def mkdir(self, path):
        path.strip()
        path.rstrip('/')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def image_count(self, name_dir):
        image_files = []
        for file in glob.glob(os.path.join(name_dir, "*")):
            image_files.append(file)
        return len(image_files)

    def save_known_image(self, frame, name, max_images=10):
        name_dir = os.path.join(self.save_image_dir, str(name))
        self.mkdir(name_dir)
        image_counts = self.image_count(name_dir)

        if image_counts >= max_images:
            print("Already saved %d images for %s" % (image_counts, name))
            return

        # use number of images in folder as id
        image_id = image_counts
        save_image = os.path.join(name_dir, str(image_id) + ".jpg")
        cv2.imwrite(save_image, frame)

    def check_face_distance(self, list_encoding, known_image):
        known_face_encoding = [encoding.values()[0] for encoding in list_encoding]
        for test_encoding in list_encoding:
            face_distances = face_recognition.face_distance(known_face_encoding, test_encoding.values()[0])
            if np.where(face_distances > 0.6):
                print(face_distances)
