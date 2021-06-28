import os

import hand_tracker as ht
import model_trainer as md

tracker = ht.HandTracker(training=False, min_detection_confidence=0.7)

tracker.camera_capture(source='assets/test/head.mp4')