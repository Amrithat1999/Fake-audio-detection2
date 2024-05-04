import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Function to extract features from audio files in a folder
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return np.array([mfccs])
    except Exception as e:
        print("Error encountered while processing file:", file_path)
        return None



def runtest(example_file_path):
    # Load the model
    loaded_model = joblib.load("random_forest_model4MixedDataALl.joblib")

    # Extract features from the example file
    example_features = extract_features(example_file_path)
    if example_features is not None:
        # Get the probability of the prediction
        proba = loaded_model.predict_proba(example_features)
        proba_fake, proba_real = proba[0]
        
        # Determine the class with the maximum probability
        max_proba_class = "Real" if proba_real > proba_fake else "Fake"
        
        return f"{max_proba_class} Audio File with probability: Real {proba_real*100:.2f}%, Fake {proba_fake*100:.2f}%"
    else:
        return "Error extracting features from the example file."


