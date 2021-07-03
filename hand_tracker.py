# import time
import os
import os.path
# from pathlib import Path
import csv

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle

# from pandas import core

FRAME_COUNT = 1
FRAMES_TO_PRINT = 4
TRAINING_CSV = 'coords.csv'
PREDICTION_CSV = 'predict.csv'
PREDICTION_FILE = 'predict.txt'
LOG_FILE = 'log.txt'

class HandTracker:
    def __init__(self, static_image_mode=False, 
                 max_num_hands=1,
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5,
                 training=False):

        # Mediapipe init settings
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.training = training
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        # CSV Header
        self.csv_header = ['class']
        for frame in range(FRAME_COUNT):
            # Leave out point at the base of the wrist.
            for val in range(1,21):
                self.csv_header += ['x{}_{}'.format(val, frame), 'y{}_{}'.format(val, frame), 'z{}_{}'.format(val, frame)]
        

        # Set up CSV file
        write_file_name = TRAINING_CSV if training else PREDICTION_CSV
        if os.path.exists(write_file_name):
            os.remove(write_file_name)
        self.csv_file = open(write_file_name, mode='a', newline='')
        self.csv_writer = csv.writer(self.csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.csv_writer.writerow(self.csv_header)

        # Log file
        self.log_file = open(LOG_FILE, mode='a', newline='')

        # Set up Prediction Model
        if not self.training:
            self.f = open('sign_lang.pkl', 'rb')
            self.model = pickle.load(self.f)
            if os.path.exists(PREDICTION_FILE):
                os.remove(PREDICTION_FILE)
            self.predict_file = open(PREDICTION_FILE, mode='w+')
            self.sign_counter = 0
            self.prev_sign = ''
            self.predicted_signs = ''

    def flatten_points(self, hand_points):
        coords_row = list(np.array([[(points.x - hand_points.landmark[0].x), 
                                     (points.y - hand_points.landmark[0].y), 
                                      points.z] 
                                      for points in hand_points.landmark[1:]]).flatten())
        # Get rid of scientific notation
        coords_row = ['{:.15f}'.format(var) for var in coords_row]
        return coords_row

    def points_to_csv(self, sign, coords_row):
        coords_row.insert(0, sign)
        self.csv_writer.writerow(coords_row)

    def predict_sign(self, coords_row):
        np.set_printoptions(suppress=True)
        X = pd.DataFrame([coords_row])
        sign_class = self.model.predict(X)[0]
        self.points_to_csv(sign_class, coords_row)
        sign_prob = self.model.predict_proba(X)[0]

        # Only print prediction after certain number of frames
        if self.prev_sign != sign_class:
            self.sign_counter = 0
        else:
            self.sign_counter += 1
            if self.sign_counter == FRAMES_TO_PRINT:
                print(sign_class, end='')
                self.predicted_signs += sign_class
        self.prev_sign = sign_class

    def camera_capture(self, sign="", source=0):
        cap = cv2.VideoCapture(source)
        cap.set(3, 640)
        cap.set(4, 480)

        with self.mp_hands.Hands(
                min_detection_confidence=self.min_detection_confidence,
                max_num_hands=self.max_num_hands) as hands:
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    cap.release()
                    cv2.destroyAllWindows()
                    break    

                # Flip image, convert to RGB, and draw landmarks.
                if source == 0:
                    img = cv2.flip(img, 1)
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.results = hands.process(imgRGB)
                if self.results.multi_hand_landmarks:
                    for hand_landmarks in self.results.multi_hand_landmarks:    
                        self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                if self.results.multi_hand_landmarks:
                    coords = self.results.multi_hand_landmarks
                    for hand in coords:
                        coords_row = self.flatten_points(hand)
                        if self.training:
                            self.points_to_csv(sign, coords_row)
                        else:
                            self.predict_sign(coords_row)

                cv2.imshow("Image", img)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    if not self.training:
                        self.f.close()
                    break

        if self.training:
            return self.dataframe, self.dataframe_norot, self.sign_list, self.sign_list_norot
        else:
            self.predict_file.write(self.predicted_signs)
            self.predict_file.write("\n")
            self.sign_counter = 0
            self.prev_sign = ''
            self.predicted_signs = ''
            print()