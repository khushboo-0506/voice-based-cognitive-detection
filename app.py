from flask import Flask, request, render_template
import os
import pickle
import speech_recognition as sr
import numpy as np
import librosa
import parselmouth
import re
from parselmouth.praat import call

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Set UPLOAD_FOLDER using a relative, OS-independent path
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "Could not understand audio"
        except sr.RequestError as e:
            text = f"Could not request results; {e}"
    return text

def extract_features(audio_path, transcript):
    try:
        y, sr_val = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr_val)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr_val)

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr_val)
        pitches = pitches[pitches > 0]
        mean_pitch = np.mean(pitches)
        pitch_variability = np.std(pitches)

        intervals = librosa.effects.split(y, top_db=30)
        speech_segments = sum((end - start) for start, end in intervals) / sr_val
        pause_count = len(intervals) - 1

        transcript = transcript.lower()
        words = transcript.split()
        word_count = len(words)
        sentence_count = len(re.findall(r'[.!?]', transcript)) or 1
        hesitation_words = ['uh', 'um', 'er', 'ah', 'eh']
        hesitation_count = sum(transcript.count(hw) for hw in hesitation_words)
        pauses_per_sentence = pause_count / sentence_count
        hesitations = hesitation_count
        common_subs = ['thing', 'stuff', 'something', 'thingy', 'whatever']
        word_recall_issues = sum(transcript.count(w) for w in common_subs)
        generic_words = ['thing', 'object', 'place', 'person', 'stuff']
        naming_issues = sum(transcript.count(w) for w in generic_words)
        completed_sentences = re.findall(r'\b[^.?!]*[.?!]', transcript)
        percent_complete_sentences = len(completed_sentences) / sentence_count
        speech_rate = word_count / duration if duration > 0 else 0

        sound = parselmouth.Sound(audio_path)
        formant = call(sound, "To Formant (burg)", 0.0, 5, 5000, 0.025, 50)
        formant_frequencies = [call(formant, "Get value at time", i, 0.0, 'Hertz', 'Linear') for i in [1, 2]]
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 300)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        return {
            'duration': duration,
            'tempo': tempo,
            'mean_pitch': float(mean_pitch),
            'pitch_variability': float(pitch_variability),
            'pauses': pause_count,
            'pauses_per_sentence': pauses_per_sentence,
            'hesitations': hesitations,
            'word_recall_issues': word_recall_issues,
            'naming_issues': naming_issues,
            'sentence_completion': round(percent_complete_sentences, 2),
            'speech_rate': speech_rate,
            'formant_frequencies': formant_frequencies,
            'jitter': jitter,
            'shimmer': shimmer
        }
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return {}

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return "No audio file uploaded", 400

    file = request.files['audio']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    transcript = transcribe_audio(filepath)
    features = extract_features(filepath, transcript)

    if not features:
        return "Error extracting features from audio", 500

    # Prepare feature vector for prediction
    feature_vector = np.array([
        features['speech_rate'],
        features['mean_pitch'],
        features['hesitations'],
        features['pauses'],
        features['jitter'],
        features['shimmer']
    ]).reshape(1, -1)

    prediction = model.predict(feature_vector)[0]
    result = "Likely Cognitive Decline" if prediction == 1 else "Likely Healthy"

    return render_template('result.html', transcript=transcript, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)