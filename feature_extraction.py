import librosa
import librosa.display
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from audio_processing import reduce_noise

sr = 32000

def feature_extractor(sound_path):
    header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' de_chroma_stft de_rmse de_spectral_centroid de_spectral_bandwidth de_rolloff de_zero_crossing_rate'
    for i in range(1, 21):
        header += f' de_mfcc{i}'
    header = header.split()
    
    # Create or clear the CSV file
    with open('extracted_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    y, y_sr = librosa.load(sound_path, mono=True)
    features = extract_features(y, y_sr)
    
    y_denoise = reduce_noise(y, y_sr)
    features_denoise = extract_features(y_denoise, y_sr)
    
    features_combined = features + features_denoise
    
    # Save to CSV
    with open('extracted_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(features_combined)
    
    data = pd.read_csv('extracted_data.csv')
    return data.values

def extract_features(y, sr):
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    features = [
        np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw),
        np.mean(rolloff), np.mean(zcr)
    ]
    
    for e in mfcc:
        features.append(np.mean(e))
    
    return features
