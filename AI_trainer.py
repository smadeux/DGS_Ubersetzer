import time
import sys

import pandas as pd

import hand_tracker as ht
import model_trainer as md
import self_learn

# tracker = ht.HandTracker(training=True, min_detection_confidence=0.7)

def translate_video(file_name):
    print("translate")
    tracker = ht.HandTracker(training=False, min_detection_confidence=0.7)
    pred_word = tracker.camera_capture(source=file_name)

    true_word = input('What was signed in the video? (Use capital letters and combine double letters into one): ')
    self_learn.parse_feedback_file(pred_word, true_word)

# if __name__ == '__main__':
#     main()

# s = input('Use webcam for training? y/n: ')
# if s == 'y':
#     while True:
#         sign = input("Type the sign and press enter (quit to exit): ")
#         if sign == 'quit':
#             # tracker.normalize_data()
#             md.train_model('coords.csv')
#             break
        
#         tracker.camera_capture(sign)
# elif s == 't':
#     md.train_model('coords.csv')
# else:  
#     # Use reference material
#     start_time = time.perf_counter()
#     for s in ht.letter_list:
#         temp_df, temp_df_norot, temp_sign_list, temp_sign_list_norot = tracker.camera_capture(sign=s, source='assets/{}.mp4'.format(s))
#         print('Sign {} Complete'.format(s))

#     tracker.destroy()
#     end_coll = time.perf_counter()
#     # Train model
#     sys.stdout = open('log.txt', "a")
#     # md.train_model('coords_norot.csv')
#     md.train_model('coords.csv')
#     # md.train_model('coords_norm.csv')
#     # md.train_model('coords_norm_norot.csv')

#     end_time = time.perf_counter()

#     print('Data collection took {}m {}s'.format(int((end_coll - start_time)/60), 
#                                                 int((end_coll - start_time)%60)))
#     print('Training took {}m {}s'.format(int((end_time - end_coll)/60), 
#                                         int((end_time - end_coll)%60)))
#     print('Total time {}m {}s'.format(int((end_time - start_time)/60), 
#                                     int((end_time - start_time)%60)))

