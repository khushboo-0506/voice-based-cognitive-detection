import os
import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import speech_recognition as sr
import pickle
import re
from datetime import datetime
import gc  # Import garbage collector

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"API Error: {e}"

def extract_features(audio_path, transcript):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        mean_pitch = np.mean(pitches[pitches > 0])
        pitch_variability = np.std(pitches[pitches > 0])
        intervals = librosa.effects.split(y, top_db=30)
        speech_segments = sum((end - start) for start, end in intervals) / sr
        silence_duration = duration - speech_segments
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
        formant_frequencies = [call(formant, "Get value at time", 1, 0.0, 'Hertz', 'Linear'),
                                call(formant, "Get value at time", 2, 0.0, 'Hertz', 'Linear')]
        pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 300)
        jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
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

audio_folder = os.path.join(os.getcwd(), 'uploads')
feature_list = []
transcripts = {}

for file in os.listdir(audio_folder):
    if file.endswith(".wav") or file.endswith(".mp3"):
        path = os.path.join(audio_folder, file)
        transcript = transcribe_audio(path)
        print(f"Transcription for {file}:\n{transcript}\n{'-'*60}")
        transcripts[file] = transcript
        features = extract_features(path, transcript)
        features['file'] = file
        feature_list.append(features)
        gc.collect()  # Force garbage collection

df = pd.DataFrame(feature_list)
df.fillna(0, inplace=True)
X = df[['speech_rate', 'mean_pitch', 'hesitations', 'pauses', 'jitter', 'shimmer']]
kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
dbscan = DBSCAN(eps=0.5, min_samples=2).fit(X)
df['kmeans_cluster'] = kmeans.labels_
df['dbscan_cluster'] = dbscan.labels_

# Save the model as model.pkl
with open("model.pkl", "wb") as model_file:
    pickle.dump(kmeans, model_file)

# Optional Visualization (can be commented out for deployment)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('KMeans Clustering')
plt.show()