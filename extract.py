#====================== Extracting Features Using Librosa ==================

"""
In order to classify new audio files not included in the original dataset, we need to be able to 
extract the features. The code below uses the Librosa Python library to do so.
"""

import librosa
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler   
import joblib                    

def extract_features(filename):
    
    cols = ['chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
       'spectral_centroid_mean', 'spectral_centroid_var',
    #    'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
    #    'rolloff_var', 
       'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'tempo',
       'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean',
       'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
       'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean',
       'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
       'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
       'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
       'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
       'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var']
    
    feature_matrix = pd.DataFrame(columns=cols)
    
    y, sr = librosa.load(filename)
    
    #Chroma Stft
    feature_matrix['chroma_stft_mean'] = [np.mean(librosa.feature.chroma_stft(y=y, sr=sr))]
    feature_matrix['chroma_stft_var'] = [np.var(librosa.feature.chroma_stft(y=y, sr=sr))]
    
    #RMS
    feature_matrix['rms_mean'] = [np.mean(librosa.feature.rms(y=y))]
    feature_matrix['rms_var'] = [np.var(librosa.feature.rms(y=y))]
    
    #Spectral Centroid
    feature_matrix['spectral_centroid_mean'] = [np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))]
    feature_matrix['spectral_centroid_var'] = [np.var(librosa.feature.spectral_centroid(y=y, sr=sr))]
    
    # #Spectral Bandwidth
    # feature_matrix['spectral_bandwidth_mean'] = [np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))]
    # feature_matrix['spectral_bandwidth_var'] = [np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))]
    
    # #Rolloff
    # feature_matrix['rolloff_mean'] = [np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))]
    # feature_matrix['rolloff_var'] = [np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))]
    
    #Zero Crossing Rate
    feature_matrix['zero_crossing_rate_mean'] = [np.mean(librosa.feature.zero_crossing_rate(y=y,frame_length=2048, hop_length=512, center=True))]
    feature_matrix['zero_crossing_rate_var'] = [np.var(librosa.feature.zero_crossing_rate(y=y,frame_length=2048, hop_length=512, center=True))]

    #Tempo
    feature_matrix['tempo'] = [librosa.feature.tempo(y=y, sr=sr, onset_envelope=None, tg=None, hop_length=512, start_bpm=120, std_bpm=1.0, ac_size=8.0, max_tempo=320.0, prior=None)]
    
    #mfcc
    mfccs = librosa.feature.mfcc(y=y, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0)
    # print("mfccs", mfccs)
    mfcc_means = []
    mfcc_vars = []
    
    j = 1
        
    for i in mfccs:
        mfcc_means.append([np.mean(i)])
        mfcc_vars.append([np.mean(i)])
        feature_matrix[f'mfcc{j}_mean'] = np.mean(i)
        feature_matrix[f'mfcc{j}_var'] = np.var(i)
        j+=1
        
    #scale data and convert back to dataframe    
    
    scaler = joblib.load('scaler_fit')
    # scaler = StandardScaler()
    feature_matrix = scaler.transform(feature_matrix)
    feature_matrix = pd.DataFrame(feature_matrix, columns=cols)
        
    return feature_matrix