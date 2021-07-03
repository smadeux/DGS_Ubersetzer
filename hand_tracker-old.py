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

from pandas import core

FRAME_COUNT = 1

# TODO - Only delete CSV files when training

class HandTrackel:

    def __init__(self, static_image_mode=False, 
                 max_num_hands=1,
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5,
                 training=False):
                 #height = 810, width = 1080):
        
        # if os.path.exists('coords.csv'):
        #     os.remove('coords.csv')
        # if os.path.exists('coords_norot.csv'):
        #     os.remove('coords_norot.csv')
        # if os.path.exists('coords_norm.csv'):
        #     os.remove('coords_norm.csv')
        # if os.path.exists('coords_norm_norot.csv'):
        #     os.remove('coords_norm_norot.csv')
        
        # Mediapipe init settings
        # self.static_image_mode = static_image_mode
        # self.max_num_hands = max_num_hands
        # self.min_detection_confidence = min_detection_confidence
        # self.min_tracking_confidence = min_tracking_confidence
        # self.mp_hands = mp.solutions.hands
        # self.mp_draw = mp.solutions.drawing_utils
        self.frames = [[] for i in range(FRAME_COUNT)]
        # self.dataframe = pd.DataFrame
        # self.dataframe_norot = pd.DataFrame
        # self.sign_list = []
        # self.sign_list_norot = []
        # self.sign_counter = 0
        # self.prev_sign = ''
        # self.predicted_sign = ''
        # self.header_points = ['class']
        # for frame in range(FRAME_COUNT):
        #     # Leave out point at the base of the wrist.
        #     for val in range(1,21):
        #         self.header_points += ['x{}_{}'.format(val, frame), 'y{}_{}'.format(val, frame), 'z{}_{}'.format(val, frame)]
        # self.training = training
        # if not self.training:
        #     self.f = open('sign_lang.pkl', 'rb')
        #     self.model = pickle.load(self.f)

        # self.file_names = ['coords.csv','coords_norot.csv','coords_norm.csv','coords_norm_norot.csv']

        # for f in self.file_names:
        #     if os.path.exists(f):
        #         os.remove(f)
        #     with open(f, mode='a', newline='') as csv_file:
        #         csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #         csv_writer.writerow(self.header_points)
            
        # if os.path.exists('log.txt'):
        #     os.remove('log.txt')

        # self.csv_file = open('coords.csv', mode='a', newline='')
        # self.csv_file_norot = open('coords_norot.csv', mode='a', newline='')
        # self.csv_writer = csv.writer(self.csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # self.csv_writer_norot = csv.writer(self.csv_file_norot, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # self.pred_file = open('predict.csv', mode='a', newline='')
        # self.pred_writer = csv.writer(self.pred_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # def normalize_data(self):
        # print("In normalize_data begin: ")
        # self.dataframe.info()
        for col in self.header_points[1:]:
            column = self.dataframe[col]
            column_norm = [((var-column.min())/(column.max()-column.min())) for var in column]
            self.dataframe.loc[:,col] = column_norm
        
        # print('After normalization: ')
        # self.dataframe.info()
        norm_list = self.dataframe.values.tolist()
        # print('After tolist: ')
        # self.dataframe.info()
        # print("coords_norm: ", len(norm_list))
        with open(self.file_names[2], mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for sign, row in zip(self.sign_list, norm_list):
                row.insert(0, sign)
                csv_writer.writerow(row)

        for col in self.header_points[1:]:
            column = self.dataframe_norot[col]
            column_norm = [((var-column.min())/(column.max()-column.min())) for var in column]
            self.dataframe_norot.loc[:,col] = column_norm
        
        # self.dataframe_norot.info()
        norm_list = self.dataframe_norot.values.tolist()
        # print("coords_norm_norot: ", len(norm_list))
        with open(self.file_names[3], mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for sign, row in zip(self.sign_list_norot, norm_list):
                row.insert(0, sign)
                csv_writer.writerow(row)

    # def header_to_csv(self):
        self.dataframe = pd.DataFrame(columns=self.header_points[1:])
        self.dataframe_norot = pd.DataFrame(columns=self.header_points[1:])


    def points_to_csv(self, sign, hand_points):
        coords_row = list(np.array([[(points.x - hand_points.landmark[0].x), 
                                     (points.y - hand_points.landmark[0].y), 
                                      points.z] 
                                      for points in hand_points.landmark[1:]]).flatten())
        coords_row_rotate = list(np.array([[(points.z), 
                                            (points.y - hand_points.landmark[0].y), 
                                            (points.x - hand_points.landmark[0].x)] 
                                             for points in hand_points.landmark[1:]]).flatten())

        # Dataframes for later normalization
        df_series = pd.Series(coords_row, index=self.dataframe.columns)
        df_series_rotate = pd.Series(coords_row_rotate, index=self.dataframe.columns)
        self.sign_list.append(sign)
        self.sign_list.append(sign)
        self.sign_list_norot.append(sign)
        self.dataframe = self.dataframe.append(df_series, ignore_index=True)
        self.dataframe = self.dataframe.append(df_series_rotate, ignore_index=True)
        self.dataframe_norot = self.dataframe_norot.append(df_series, ignore_index=True)
        # print("DaTaFrAmE: ")
        # self.dataframe.info()

        # Get rid of scientific notation
        coords_row = ['{:.15f}'.format(var) for var in coords_row]
        coords_row_rotate = ['{:.15f}'.format(var) for var in coords_row_rotate]

        coords_row.insert(0, sign)
        coords_row_rotate.insert(0, sign)

        # self.csv_writer.writerow(coords_row)
        # self.csv_writer.writerow(coords_row_rotate)
        # self.csv_writer_norot.writerow(coords_row)
        self.pred_writer.writerow(coords_row)
        
    def predict_sign(self, hand_points):
        coords_row = list(np.array([[(points.x - hand_points.landmark[0].x), 
                                     (points.y - hand_points.landmark[0].y), 
                                      points.z] 
                                      for points in hand_points.landmark[1:]]).flatten())

        # Get rid of scientific notation
        coords_row = ['{:.15f}'.format(var) for var in coords_row]

        # Predict sign
        np.set_printoptions(suppress=True)
        X = pd.DataFrame([coords_row])
        sign_class = self.model.predict(X)[0]
        self.points_to_csv(sign_class, hand_points)
        sign_prob = self.model.predict_proba(X)[0]
        if self.prev_sign != sign_class:
            self.sign_counter = 0
        else:
            if self.sign_counter == 5:
                print(sign_class)
                self.predicted_sign += sign_class
            self.sign_counter += 1
        self.prev_sign = sign_class

    def camera_capture(self, sign="", save_csv=True, draw=True, 
                       num_hands=1, source=0):
        cap = cv2.VideoCapture(source)
        cap.set(3, 640)
        cap.set(4, 480)

        first_frame = True

        # Create file for tracking points and set up header.
        if self.training:
            csv_name = time.ctime(time.time())
            self.header_to_csv()

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
                
                # self.frames.insert(0, [])
                # self.frames.pop(-1)
                if self.results.multi_hand_landmarks:
                    coords = self.results.multi_hand_landmarks
                    for hand in coords:
                        if self.training:
                            self.points_to_csv(sign, hand)
                        else:
                            self.predict_sign(hand)

                cv2.imshow("Image", img)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    self.cleanup_data()
                    if not self.training:
                        self.f.close()
                    break

        if self.training:
            self.cleanup_data()
            return self.dataframe, self.dataframe_norot, self.sign_list, self.sign_list_norot
        else:
            file = open("predicted.txt", "w")
            file.write(self.predicted_sign)
    
    def cleanup_data(self):
        self.normalize_data()
        self.sign_list.clear()
        self.sign_list_norot.clear()

    def destroy(self):
        self.csv_file.close()
        self.csv_file_norot.close()

#Non class code
letter_map = {
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

letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z']