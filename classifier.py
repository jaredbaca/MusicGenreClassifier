import pickle
from extract import extract_features
import tkinter as tk
from tkinter import filedialog
import customtkinter
import pyaudio  
import sys
import pygame
import os

"""""
This file separates out the classifier code from the Jupyter Notebook so that it can be packaged as a standalone application. 
The trained model and scaler files are imported from the Jupyter notebook and used to classify new files selected by the user.
This code also provides a GUI using tkinter.

While not strictly necessary for the data science portion of this project, this app presents the finished product in a user friendly way.
"""""

filepath = ""

# Pygame is used to playback audio files
pygame.mixer.init()

# Function for selecting the file to be classified
def browseFiles():
    global filepath
    filepath = filedialog.askopenfilename(initialdir=".", title="Select Audio File",filetypes=("Audio files",".mp3 .wav .ogg"))
    filename = filepath.split('/')[-1]
    
    label_file_explorer.configure(text="File Opened: "+filename)
    button_play.pack(pady=10)
    button_stop.pack(pady=10)
    button_genre.pack(pady=10)
    result.pack(pady=50)
    result.configure(text="")
    accuracy.configure(text="")

# Run the classifier on user selected audio file    
def detectGenre():
    with open('model.pkl', 'rb') as f:
        rf = pickle.load(f)
        
    features = extract_features(os.path.abspath(filepath))
    genre = rf.predict(features)
    result.configure(text=genre[0].capitalize())
    accuracy.pack()
    accuracy.configure(text="Random Forest Classifier \n Accuracy 70%")

# Play and stop the audio file   
def play():
    pygame.mixer.music.load(os.path.abspath(filepath))
    pygame.mixer.music.play()
    
def stop():
    pygame.mixer.music.stop()
 

# =============== UI Created With Custom Tkinter ===============

# System Settings
customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("dark-blue")

# App Frame
app = customtkinter.CTk()
app.geometry("720x480")
app.title("Music Genre Classifier")

# Adding UI Elements
title = customtkinter.CTkLabel(app, text="Upload an Audio File")
title.pack() 

# Progress Bar
progressbar = customtkinter.CTkProgressBar(app, 
                                           orientation="horizontal",
                                           mode="indeterminate",
                                           indeterminate_speed=1
                                           )

# File Dialog
label_file_explorer = customtkinter.CTkLabel(app,
                            text="File Explorer using Tkinter",
                            width= 100, height=4,
                            # fg="blue"
                            )

# Upload Button
button_explore = customtkinter.CTkButton(app,
                        text= "Browse Files",
                        command = browseFiles)

# Genre Button
button_genre = customtkinter.CTkButton(app,
                        text= "Detect Genre",
                        command = detectGenre)

# Play Button
button_play = customtkinter.CTkButton(app,
                                      text="Play",
                                      command=play)

# Stop Button
button_stop = customtkinter.CTkButton(app,
                                      text="Stop",
                                      command=stop)

# Result 
result = customtkinter.CTkLabel(app,
                  text="",
                  font=('Helvetica', 64),  
                  height=5, width = 52)

# Accuracy 
accuracy = customtkinter.CTkLabel(app,
                  text="",
                  font=('Helvetica', 14),  
                  height=5, width = 52)

# Place elements
label_file_explorer.pack(pady=10)
button_explore.pack(pady=10)



#Run app
app.mainloop()