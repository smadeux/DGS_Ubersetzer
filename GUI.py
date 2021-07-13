import tkinter as tk
from tkinter import filedialog as fd

import AI_trainer
import hand_tracker as ht

### 
# TODO
# 1. Output test results
# 2. Make GUI Look nicer
# 3. 
###

def btn_sandbox_mode():
    AI_trainer.sandbox_mode()

def btn_translate_video():
    f = fd.askopenfile(filetypes=[("Videos", "*.mp4"),
                                  ("Videos", "*.avi")])
    pred_word = AI_trainer.translate_video(f.name)
    print(pred_word)
    message_popup("Here is what the applicaiton predicted: {}".format(pred_word))

def btn_translate_video_learn():
    f = fd.askopenfile(filetypes=[("Videos", "*.mp4"),
                                  ("Videos", "*.avi")])
    AI_trainer.translate_video(f.name, True)
    message_popup("Self learning complete\nNew training material saved")

def btn_train_webcam():
    result = AI_trainer.webcam_train()
    message_popup("Model Retrained\n{}".format(result))

def btn_train_csv():
    result = AI_trainer.train_model_from_csv(ht.TRAINING_CSV)
    message_popup("Model Retrained\n{}".format(result))

def message_popup(message):
    popup = tk.Tk()
    popup.wm_title("DGS_Übersetzer")
    label = tk.Label(popup, text=message)
    label.pack(side="top", fill="x", pady=10, padx=10)
    B1 = tk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()

def main():
    window = tk.Tk()
    window.title("DGS_Übersetzer")

    sandbox_mode = tk.Button(text="Translate Signs", command=btn_sandbox_mode)
    sandbox_mode.pack()

    translate_video = tk.Button(text="Translate Video", command=btn_translate_video)
    translate_video.pack()

    translate_video_learn = tk.Button(text="Translate Video w\ Self Learning", command=btn_translate_video_learn)
    translate_video_learn.pack()

    train_webcam = tk.Button(text="Train Model with Webcam", command=btn_train_webcam)
    train_webcam.pack()

    train_csv = tk.Button(text="Train Model with Gathered Data", command=btn_train_csv)
    train_csv.pack()

    quit_btn = tk.Button(text="Quit", command=quit)
    quit_btn.pack()

    window.mainloop()

if __name__ == '__main__':
    main()