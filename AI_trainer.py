import time
import sys

import pandas as pd
import tkinter as tk
from tkinter import simpledialog

import hand_tracker as ht
import model_trainer as md
import self_learn

def sandbox_mode():
    tracker = ht.HandTracker(training=False, min_detection_confidence=0.7)
    tracker.camera_capture(sandbox=True)

def translate_video(file_name, learn=False):
    tracker = ht.HandTracker(training=False, min_detection_confidence=0.7)
    pred_word = tracker.camera_capture(source=file_name)
    if learn:
        true_word = simpledialog.askstring(title="DGS_Übersetzer",
                                        prompt="Here is what the applicaiton predicted: {}\nWhat was actually signed in the video? (Use capital letters and combine double letters into one): ".format(pred_word))
        if true_word != "":
            self_learn.parse_feedback_file(pred_word, true_word)
    else:
        return pred_word

def webcam_train():
    tracker = ht.HandTracker(training=True, min_detection_confidence=0.7)
    while True:
        sign = simpledialog.askstring(title="DGS_Übersetzer",
                                      prompt="Type the sign and press enter (quit to exit): \n(Make sure you only use one sign at a time)")
        if sign == 'quit':
            return md.train_model(ht.TRAINING_CSV)
        
        tracker.camera_capture(sign)

def train_model_from_csv(file_name):
    return md.train_model(file_name)

def initial_training():
    tracker = ht.HandTracker(training=True, min_detection_confidence=0.7)
    # Use reference material
    start_time = time.perf_counter()
    for s in ht.letter_list:
        tracker.camera_capture(sign=s, source='assets/{}.mp4'.format(s))
        print('Sign {} Complete'.format(s))

    tracker.destroy()
    end_coll = time.perf_counter()
    # Train model)
    result = md.train_model(ht.TRAINING_CSV)

    end_time = time.perf_counter()

    print('Data collection took {}m {}s'.format(int((end_coll - start_time)/60), 
                                                int((end_coll - start_time)%60)))
    print('Training took {}m {}s'.format(int((end_time - end_coll)/60), 
                                        int((end_time - end_coll)%60)))
    print('Total time {}m {}s'.format(int((end_time - start_time)/60), 
                                    int((end_time - start_time)%60)))

    return result

