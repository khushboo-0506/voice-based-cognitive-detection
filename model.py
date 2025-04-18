# Imports
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import speech_recognition as sr
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import parselmouth
from parselmouth.praat import call
from transformers import pipeline
from pydub import AudioSegment
import pandas as pd
import re
import pickle



def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return f"API Error: {e}"
    

def extract_features(audio_path, transcript):
    y, sr = librosa.load(audio_path, sr=None)

    # === 1. Duration
    duration = librosa.get_duration(y=y, sr=sr)

    # === 2. Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # === 3. Pitch Features
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    mean_pitch = np.mean(pitches[pitches > 0])
    pitch_variability = np.std(pitches[pitches > 0])

    # === 4. Pauses
    intervals = librosa.effects.split(y, top_db=30)
    speech_segments = sum((end - start) for start, end in intervals) / sr
    silence_duration = duration - speech_segments
    pause_count = len(intervals) - 1  # gaps between speech parts

    # === 5. Transcript-based features ===
    transcript = transcript.lower()
    words = transcript.split()
    word_count = len(words)
    sentence_count = len(re.findall(r'[.!?]', transcript)) or 1
    hesitation_words = ['uh', 'um', 'er', 'ah', 'eh']
    hesitation_count = sum(transcript.count(hw) for hw in hesitation_words)

    # === 6. Pauses per sentence
    pauses_per_sentence = pause_count / sentence_count

    # === 7. Hesitations
    hesitations = hesitation_count

    # === 8. Word Recall Issues (very basic check)
    common_subs = ['thing', 'stuff', 'something', 'thingy', 'whatever']
    word_recall_issues = sum(transcript.count(w) for w in common_subs)

    # === 9. Naming & Word-Association Tasks
    generic_words = ['thing', 'object', 'place', 'person', 'stuff']
    naming_issues = sum(transcript.count(w) for w in generic_words)

    # === 10. Sentence Completion
    completed_sentences = re.findall(r'\b[^.?!]*[.?!]', transcript)
    percent_complete_sentences = len(completed_sentences) / sentence_count

    # === 11. Speech Rate
    speech_rate = word_count / duration if duration > 0 else 0

    # === 12. Formant Frequencies
    sound = parselmouth.Sound(audio_path)
    formant = call(sound, "To Formant (burg)", 0.0, 5, 5000, 0.025, 50)
    formant_frequencies = [call(formant, "Get value at time", 1, 0.0, 'Hertz', 'Linear'),
                            call(formant, "Get value at time", 2, 0.0, 'Hertz', 'Linear')]

    # === 13. Jitter and Shimmer
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

audio_folder = r"C:\Users\raikh\OneDrive\Desktop\deploy\uploads"  # Replace with your folder path
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
        


df = pd.DataFrame(feature_list)
X = df[['speech_rate', 'mean_pitch', 'hesitations', 'pauses', 'jitter', 'shimmer']]
kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
dbscan = DBSCAN(eps=0.5, min_samples=2).fit(X)
df['kmeans_cluster'] = kmeans.labels_
df['dbscan_cluster'] = dbscan.labels_

pickle.dump(kmeans,open("model.pkl","wb"))

