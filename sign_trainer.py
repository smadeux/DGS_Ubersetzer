import os
import time

import hand_tracker as ht
import model_trainer as md

tracker = ht.HandTracker(training=True, min_detection_confidence=0.7)

# if os.path.exists('coords.csv'):
#     os.remove('coords.csv')

s = input('Use webcam for training? y/n: ')
if s == 'y':
    while True:
        sign = input("Type the sign and press enter (quit to exit): ")
        if sign == 'quit':
            break
        
        tracker.camera_capture(sign)
else:  
    # Use reference material
    start_time = time.gmtime()
    tracker.camera_capture(sign='A', source='assets/A.mp4')
    tracker.camera_capture(sign='B', source='assets/B.mp4')
    tracker.camera_capture(sign='C', source='assets/C.mp4')
    tracker.camera_capture(sign='D', source='assets/D.mp4')
    tracker.camera_capture(sign='E', source='assets/E.mp4')
    tracker.camera_capture(sign='F', source='assets/F.mp4')
    tracker.camera_capture(sign='G', source='assets/G.mp4')
    tracker.camera_capture(sign='H', source='assets/H.mp4')
    tracker.camera_capture(sign='I', source='assets/I.mp4')
    # tracker.camera_capture(sign='J', source='assets/J.mp4') Messing things up
    end_coll = time.gmtime()


# Train model
md.train_model('coords.csv')

end_time = time.gmtime()

print('Data collection took {}m {}s'.format((end_coll.tm_min - start_time.tm_min), 
                                            (end_coll.tm_sec - start_time.tm_sec)))
print('Training took {}m {}s'.format((end_time.tm_min - end_coll.tm_min), 
                                     (end_time.tm_sec - end_coll.tm_sec)))
print('Total time {}m {}s'.format((end_time.tm_min - start_time.tm_min), 
                                  (end_time.tm_sec - start_time.tm_sec)))