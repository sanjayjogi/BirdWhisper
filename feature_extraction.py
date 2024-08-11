import librosa
import numpy as np
import pandas as pd
from audio_processing import reduce_noise

sr = 32000

def feature_extractor(sound_path):
    try:
        header = [
            'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate'
        ] + [f'mfcc{i}' for i in range(1, 21)] + [
            'de_chroma_stft', 'de_rmse', 'de_spectral_centroid', 'de_spectral_bandwidth', 'de_rolloff', 'de_zero_crossing_rate'
        ] + [f'de_mfcc{i}' for i in range(1, 21)]
        
        y, y_sr = librosa.load(sound_path, sr=sr, mono=True)
        
        if y is None or len(y) == 0:
            raise ValueError("Loaded audio is empty")
        
        features = extract_features(y, y_sr)
        
        y_denoise = reduce_noise(y, y_sr)
        features_denoise = extract_features(y_denoise, y_sr)
        
        features_combined = features + features_denoise
        
        return np.array(features_combined).reshape(1, -1)
    except Exception as e:
        print(f"Error extracting features from {sound_path}: {e}")
        return np.zeros((1, len(header)))

def extract_features(y, sr):
    try:
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
        
        features.extend(np.mean(mfcc, axis=1))
        
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return [0] * 26  

