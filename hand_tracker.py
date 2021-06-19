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

FRAME_COUNT = 1

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
        self.frames = [[] for i in range(FRAME_COUNT)]
        self.dataframe = pd.DataFrame
        self.sign_list = []
        self.sign_counter = 0
        self.prev_sign = ''
        self.training = training
        if not self.training:
            self.f = open('sign_lang.pkl', 'rb')
            self.model = pickle.load(self.f)

        self.letter_map = {
            0: 'A',
            1: 'B',
            2: 'C' ,
            3: 'D',
            4: 'E',
            5: 'F',
            6: 'G',
            7: 'H',
            8: 'I',
            9: 'J'
        }

    def normalize_data(self):
        output_list = []
        # columns = ['horiz_mov_{}'.format(val) for val in range(FRAME_COUNT-1)]
        # for frame in range(FRAME_COUNT):
        #     for val in range(1,21):
        #         columns += ['x{}_{}'.format(val, frame), 'y{}_{}'.format(val, frame), 'z{}_{}'.format(val, frame)]
        # dataframe = pd.read_csv('coords.csv', names=columns, header=0)
        array = self.dataframe.values
        array_flat = array.flatten()
        max_data = max(array_flat)
        min_data = min(array_flat)
        print(array)
        for row in array:
            for val in row:
                val = ((val - min_data)/(max_data - min_data))
            output_list.append(row.tolist())

        assert(len(output_list) == len(self.sign_list))
        print(output_list)

    def camera_capture(self, sign="", save_csv=True, draw=True, 
                       num_hands=1, source=0):
        cap = cv2.VideoCapture(source)
        cap.set(3, 640)
        cap.set(4, 480)

        first_frame = True

        # Create file for tracking points and set up header.
        if self.training:
            csv_name = time.ctime(time.time())
            if(not os.path.exists('coords.csv')):
                landmarks = ['class']
                landmarks += ['horiz_mov_{}'.format(val) for val in range(FRAME_COUNT-1)]
                for frame in range(FRAME_COUNT):
                    for val in range(1,21):
                        landmarks += ['x{}_{}'.format(val, frame), 'y{}_{}'.format(val, frame), 'z{}_{}'.format(val, frame)]
                
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
                if source == 0:
                    img = cv2.flip(img, 1)
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.results = hands.process(imgRGB)
                if draw and self.results.multi_hand_landmarks:
                    for hand_landmarks in self.results.multi_hand_landmarks:    
                        self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                self.frames.insert(0, [])
                self.frames.pop(-1)
                if self.results.multi_hand_landmarks:
                    coords = self.results.multi_hand_landmarks
                    for hand in coords:
                        coords_row = list(np.array([[(points.x - hand.landmark[0].x), 
                                                    (points.y - hand.landmark[0].y), 
                                                    points.z] 
                                                    for points in hand.landmark[1:]]).flatten())
                
                    self.frames[0] = coords_row
                    if self.frames[-1] != []:
                        horiz_mov = []
                        for i in range(FRAME_COUNT-1):
                            horiz_mov.append(abs(self.frames[i][11] - self.frames[i+1][11]))
                        flat_frames = [item for sublist in self.frames for item in sublist]
                        for i, point in enumerate(horiz_mov):
                            flat_frames.insert(i, point)
                        # for number in flat_frames:
                        #     number = f"{number:15f}"
                        # Normalize the data.
                        max_data = max(flat_frames)
                        min_data = min(flat_frames)
                        flat_frames = [((var-min_data)/(max_data-min_data)) for var in flat_frames]

                        flat_frames = ['{:.15f}'.format(var) for var in flat_frames]
                
                        if self.training:
                            # Write hand coordinates to csv
                            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            flat_frames.insert(0, sign)
                            # self.dataframe.append(pd.DataFrame(flat_frames), ignore_index=True)
                            # self.sign_list.append(sign)
                            # print('Flat_frames: ', flat_frames)
                            csv_writer.writerow(flat_frames)
                        else:
                            # Predict sign
                            X = pd.DataFrame([flat_frames])
                            sign_class = self.model.predict(X)[0]
                            sign_prob = self.model.predict_proba(X)[0]
                            if self.prev_sign != sign_class:
                                self.sign_counter = 0
                            else:
                                if self.sign_counter == 4:
                                    print(sign_class, sign_prob)
                                self.sign_counter += 1
                            self.prev_sign = sign_class
                            # print(sign_class, ['{:.5f}'.format(val) for val in sign_prob])
                            # for ind, prob in enumerate(sign_prob):
                            #     if prob > 0.30:
                            #         print(self.letter_map[ind], prob)
                            #     else:
                            #         print("No Sign")
                else:
                    for frame in self.frames:
                        frame.clear()

                cv2.imshow("Image", img)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print()
                    cv2.destroyAllWindows()
                    if self.training:
                        csv_file.close()
                    else:
                        self.f.close()
                    break