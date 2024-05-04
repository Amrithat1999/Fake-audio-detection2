import os
import numpy as np
import librosa

# Function to extract audio features
def extract_features(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract features (e.g., MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    
    # Concatenate features
    features = np.vstack([mfccs, spectral_centroid, zero_crossing_rate])
    
    return features

# Example usage
audio_file = 'ai gen voice-3.mp3'  # Replace with the path to your audio file
audio_features = extract_features(audio_file)
print("Extracted audio features shape:", audio_features.shape)
