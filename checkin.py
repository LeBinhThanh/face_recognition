from __future__ import division
from model.face_rec_arcface import FaceRecognizer
from datetime import datetime
from dateutil import tz
import cv2
import time
import uuid


class CheckIn(object):
    def __init__(self, input_source, face_recognizer):
        self.face_recognizer = face_recognizer
        self.image = None
        self.predicted_name = ['Unknown']
        self.cap = cv2.VideoCapture(input_source)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        self.in_predict = True
        self.time_zone = tz.tzlocal()
        self.checked_name = []
        self.today_dir = 'checkin_result/{}'.format(datetime.now().strftime('%Y%m%d'))

    def run_camera(self):
        try:
            ret, self.image = self.cap.read()
            self.predict_name()
            # copy image then add text to that copy
            image = self.image.copy()
            cv2.rectangle(image, (0, 0), (260, 50), (255, 255, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(image, self.predicted_name[0], (10, 30), font, 1.3, (0, 128, 0), 1)

            # Display the resulting image
            cv2.imshow('Video', image)

        except Exception as e:
            print('Got exception when capture video!¥n', e)
            pass

    def predict_name(self):
        if self.image is None:
            return
        found_front_face, self.predicted_name[0] = self.face_recognizer.recognize_image(self.image)
        if found_front_face:
            self.do_check_in(self.predicted_name[0])

    def do_check_in(self, name):
        if name != 'Unknown':
            now = datetime.now(self.time_zone)
            checkin_time = now.isoformat(timespec='seconds')

            if name not in self.checked_name:
                self.checked_name.append(name)
                post_message_to_channel('{}: {}'.format(name, checkin_time))

            cv2.imwrite('{}/known/{}_{}.jpg'.format(self.today_dir, name, uuid.uuid4()), self.image)
        else:
            # save unknown person image for analysis
            cv2.imwrite('{}/unknown/{}.jpg'.format(self.today_dir, uuid.uuid4()), self.image)


if __name__ == "__main__":
    checkIn = CheckIn(input_source=int(0), face_recognizer=FaceRecognizer())

    start = time.time()
    while checkIn.cap.isOpened():
        checkIn.run_camera()
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    checkIn.cap.release()
    cv2.destroyAllWindows()
