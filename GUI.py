import tkinter as tk
from tkinter import filedialog as fd

import AI_trainer

def btn_translate_video():
    f = fd.askopenfile()
    AI_trainer.translate_video(f.name)

def btn_train_webcam():
    x=2

def btn_train_csv():
    x=2


print("Main Window")
window = tk.Tk()

translate_video = tk.Button(text="Translate Video", command=btn_translate_video)
translate_video.pack()

train_webcam = tk.Button(text="Train Model with Webcam", command=btn_train_webcam)
# train_webcam.bind("", btn_train_webcam)
train_webcam.pack()

train_csv = tk.Button(text="Train Model with CSV File", command=btn_train_csv)
# train_csv.bind("", btn_train_csv)
train_csv.pack()

window.mainloop()

# if __name__ == '__main__':
#     main()