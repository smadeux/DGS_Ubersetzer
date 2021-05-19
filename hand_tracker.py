import time
import os
import os.path
from pathlib import Path
import csv

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle

class HandTracker:

    def __init__(self, static_image_mode=False, 
                 max_num_hands=1,
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5,
                 training=False):
                 #height = 810, width = 1080):
        
        # Mediapipe init settings
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.training = training
        if not self.training:
            self.f = open('sign_lang.pkl', 'rb')
            self.model = pickle.load(self.f)

    def camera_capture(self, sign="", save_csv=True, draw=True, 
                       num_hands=1, source=0):
        cap = cv2.VideoCapture(source)
        cap.set(3, 640)
        cap.set(4, 480)

        # Create file for tracking points and set up header.
        if self.training:
            csv_name = time.ctime(time.time())
            if(not os.path.exists('coords.csv')):
                landmarks = ['class']
                for val in range(21):
                    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]
                
                csv_file = open('coords.csv', mode='a', newline='')
                csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)
                csv_file.close()
        
            csv_file = open('coords.csv', mode='a', newline='')

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
                # if self.source == 0:
                #     img = cv2.flip(img, 1)
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.results = hands.process(imgRGB)
                if draw and self.results.multi_hand_landmarks:
                    for hand_landmarks in self.results.multi_hand_landmarks:    
                        self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                if self.results.multi_hand_landmarks:
                    coords = self.results.multi_hand_landmarks
                    for hand in coords:
                        coords_row = list(np.array([[points.x, 
                                                points.y, 
                                                points.z] 
                                                for points in hand.landmark]).flatten())
                    
                    if self.training:
                        # Write hand coordinates to csv
                        coords_row.insert(0, sign)
                        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(coords_row)
                    else:
                        # Predict sign
                        X = pd.DataFrame([coords_row])
                        sign_class = self.model.predict(X)[0]
                        sign_prob = self.model.predict_proba(X)[0]
                        print(sign_class, sign_prob)

                cv2.imshow("Image", img)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    if self.training:
                        csv_file.close()
                    else:
                        self.f.close()
                    break