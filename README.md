# MusicGenreClassifier

The goal of this project is to evaluate the accuracy of several machine learning algorithms for classifying music files by genre. Some preprocessing and analysis is done on the data including feature importance, a correlation matrix, and a scatterplot showing the relationship between the two most important features.

<img width="500" alt="Screen Shot 2023-12-17 at 1 16 19 PM" src="https://github.com/jaredbaca/MusicGenreClassifier/assets/110132943/28d6bad6-ee92-4b74-b529-fa1a6910771a">

[Check out the video walkthrough here!](https://youtu.be/-yvIHN7wlec)

# Project Overview
The project is a Jupyter Notebook containing the Python code used to train and compare several classification models from the SciKit Learn library. The most accurate of these models, the Random Forest Classifier, is then used in a standalone application along with a custom built UI in the classifier.py file. Instructions for launching the Classifier application can be found in the separate README file.





## Libraries Used
- Numpy
- Pandas
- SciKit Learn
- Tkinter

## About the Dataset
The feature matrix of the GTZAN dataset contains ten primary features that represent three fundamental aspects of music: timbre, rhythm, and harmony. The features are as follows:

- Chroma STFT (Short Term Fourier Transform) - The range of pitches within the segment of music.
- RMS (Root Mean Square) – A measure of the overall loudness
- Spectral Centroid – “Center of gravity” of the sound’s frequency content
- Spectral Bandwidth – The range of variance around the spectral centroid
- Spectral Roll-Off – The frequency below which 85% of the total spectral energy lies
- Zero Crossing Rate – Rate at which the signal changes from positive to negative; represents the overall “noisiness” of the audio file
- Harmony – The individual musical pitches (frequencies) present
- Perceptr – Unclear what this value represents
- Tempo – number of beats per minute
- Mel-Frequency Cepstral Coefficients (MFCC’s) – Coefficient that provide a “compact representation of the spectral envelope” (Tzanetakis), most commonly used in speech recognition.

Many features include both a mean and variance value, and there is a total of twenty MFCC’s, bringing the overall number of features close to fifty. The response vector contains labels for ten distinct genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock. 
The spectral features proved to be the most significant in determining the genre. These include spectral centroid, spectral bandwidth, and spectral roll-off, as well as the Mel-Frequency Cepstral Coefficients. These features represent the frequency content of each audio file and correspond to the timbre or overall tone of the music. A thorough mathematical explanation of the features in the dataset is beyond the scope of this project summary, but a detailed explanation can be found in the paper Musical Genre Classification of Audio Signals by George Tzanetakis. 




