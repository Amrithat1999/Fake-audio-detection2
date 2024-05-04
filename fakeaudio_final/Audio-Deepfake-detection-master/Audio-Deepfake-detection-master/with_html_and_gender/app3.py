from flask import Flask, request, render_template, redirect, url_for
import joblib
import librosa
import numpy as np
import os
import parselmouth


app = Flask(__name__)

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return np.array([mfccs])
    except Exception as e:
        print("Error encountered while processing file:", file_path)
        return None
    


#gender
#male or female

def extract_audio_features(audio_file):
    sound = parselmouth.Sound(audio_file)

    # Extract pitch
    pitch = sound.to_pitch()
    mean_pitch = np.nanmean(pitch.selected_array['frequency'])

    # Extract intensity
    intensity = sound.to_intensity()
    mean_intensity = np.nanmean(intensity.values)

    return mean_pitch, mean_intensity

def detect_gender(audio_file_path):
    mean_pitch, mean_intensity = extract_audio_features(audio_file_path)

    # Define thresholds for pitch and intensity
    pitch_threshold = 130  # Example threshold for distinguishing male and female voices
    intensity_threshold = 50  # Example threshold for distinguishing male and female voices

    # Classify based on thresholds
    if mean_pitch > pitch_threshold and mean_intensity > intensity_threshold:
        return "Female"
    else:
        return "Male"



@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        audio_file = request.files['file']
        if not os.path.exists('choosenaudios'):
            os.makedirs('choosenaudios')
        audio_file.save(os.path.join('choosenaudios', audio_file.filename))
        loaded_model = joblib.load("random_forest_modelAudioMixed2NewMoredata2.joblib")
        example_features = extract_features(os.path.join('choosenaudios', audio_file.filename))

        # print('audio_filename$$$$$$$$$$$$$$$$$',audio_file.filename)
        # gender = detect_gender(audio_file.filename)
        gender = detect_gender(os.path.join('choosenaudios', audio_file.filename))

        # print('gender################',gender)

        if example_features is not None:
            proba = loaded_model.predict_proba(example_features)
            proba_fake, proba_real = proba[0]
            max_proba_class = "Real" if proba_real > proba_fake else "Fake"
            # result = f"{max_proba_class} Audio File with probability: Real {proba_real*100:.2f}%, Fake {proba_fake*100:.2f}%"
        else:
            max_proba_class = "Error extracting features from the audio file."
        result=f"The {gender} voice is {max_proba_class}."
        return render_template('predict.html', prediction_text=result)
        # return render_template('predict.html', prediction_text=max_proba_class)
    
    return render_template('predict.html', prediction_text='Upload an audio file to predict.')

if __name__ == '__main__':
    app.run(debug=True)



# audio_file_path = "/content/AudiomixedData2/fake/AI-voice-2.mp3"
# print(runtest(audio_file_path))

# gender = detect_gender(audio_file_path)
# print("Detected Gender:", gender)

# result=runtest(audio_file_path)

# print(f"The {gender} voice is {result}.")