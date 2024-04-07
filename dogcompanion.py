import pyaudio
import sounddevice as sd
import numpy as np
import wave
import os
import pandas as pd
import librosa
import librosa.display
import joblib
from IPython.display import display
import pyttsx3
import threading
import tkinter as tk
from tkinter import Canvas
from tkinter import Tk, PhotoImage
from PIL import Image, ImageTk

def detect_noise(duration=3, threshold=1, callback=None):
    with sd.InputStream(callback=callback):
        sd.sleep(int(duration * 1000))

def record_audio(seconds, sample_rate, chunk_size):
    frames = []
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk_size)

    for i in range(0, int(sample_rate / chunk_size * seconds)):
        data = stream.read(chunk_size)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return frames

def save_audio(frames, sample_rate):
    output_file = "recorded_audio.wav"
    wave_file = wave.open(output_file, 'wb')
    wave_file.setnchannels(1)
    wave_file.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wave_file.setframerate(sample_rate)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()
    print("Audio saved as:", output_file)

    return output_file
    

def preprocess_audio(file_path, metadata):
    print("Processing audio:", file_path)

    x, sr = librosa.load(file_path)

    # MFCC Extraction
    mfcc =librosa.feature.mfcc(y=x, sr=sr)

    mfcc_min = np.min(mfcc) 
    mfcc_mean = np.mean(mfcc)
    mfcc_max = np.max(mfcc)
    mfcc_median = np.median(mfcc)
    mfcc_std = np.std(mfcc)
    mfcc_var = np.var(mfcc)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=x, sr=sr)

    chroma_min = np.min(chroma)
    chroma_mean = np.mean(chroma)
    chroma_max = np.max(chroma)
    chroma_median = np.median(chroma)
    chroma_std = np.std(chroma)
    chroma_var = np.var(chroma)

    # Spectral
    spectral_contrast = librosa.feature.spectral_contrast(y=x, sr=sr)

    s_min = np.min(spectral_contrast)
    s_mean = np.mean(spectral_contrast)
    s_max = np.max(spectral_contrast)
    s_median = np.median(spectral_contrast)
    s_std = np.std(spectral_contrast)
    s_var = np.var(spectral_contrast)
    print(f's_min {s_min}')

    # Tonnetz
    tonnetz = np.array(librosa.feature.tonnetz(y=x, sr=sr))

    t_min = np.min(tonnetz)
    t_mean = np.mean(tonnetz)
    t_max = np.max(tonnetz)
    t_median = np.median(tonnetz)
    t_std = np.std(tonnetz)
    t_var = np.var(tonnetz)

    # melspectrogram
    melspectrogram = np.array(librosa.feature.melspectrogram(y=x, sr=sr))

    m_min = np.min(melspectrogram)
    m_mean = np.mean(melspectrogram)
    m_max = np.max(melspectrogram)
    m_median = np.median(melspectrogram)
    m_std = np.std(melspectrogram)
    m_var = np.var(melspectrogram)

    if len(metadata.columns) == 0:
        metadata['mfcc_min'] = np.nan
        metadata['mfcc_mean'] = np.nan
        metadata['mfcc_max'] = np.nan
        metadata['mfcc_median'] = np.nan
        metadata['mfcc_std'] = np.nan
        metadata['mfcc_var'] = np.nan
        metadata['chroma_min'] = np.nan
        metadata['chroma_mean'] = np.nan
        metadata['chroma_max'] = np.nan
        metadata['chroma_median'] = np.nan
        metadata['chroma_std'] = np.nan
        metadata['chroma_var'] = np.nan
        metadata['spectral_min'] = np.nan
        metadata['spectral_mean'] = np.nan
        metadata['spectral_max'] = np.nan
        metadata['spectral_median'] = np.nan
        metadata['spectral_std'] = np.nan
        metadata['spectral_var'] = np.nan
        metadata['tonnetz_min'] = np.nan
        metadata['tonnetz_mean'] = np.nan
        metadata['tonnetz_max'] = np.nan
        metadata['tonnetz_median'] = np.nan
        metadata['tonnetz_std'] = np.nan
        metadata['tonnetz_var'] = np.nan
        metadata['melspectrogram_min'] = np.nan
        metadata['melspectrogram_mean'] = np.nan
        metadata['melspectrogram_max'] = np.nan
        metadata['melspectrogram_median'] = np.nan
        metadata['melspectrogram_std'] = np.nan
        metadata['melspectrogram_var'] = np.nan

    metadata.loc[len(metadata)] = [mfcc_min, mfcc_mean, mfcc_max, mfcc_median, mfcc_std, mfcc_var, 
                                   chroma_min, chroma_mean, chroma_max, chroma_median, chroma_std, chroma_var,
                                   s_min, s_mean, s_max, s_median, s_std, s_var,
                                   t_min, t_mean, t_max, t_median, t_std, t_var,
                                   m_min, m_mean, m_max, m_median, m_std, m_var]
    

def classify_audio(model, dataframe):
    features = ['mfcc_mean', 'mfcc_min', 'mfcc_max', 'mfcc_median', 'mfcc_std', 'mfcc_var', 'chroma_mean', 'chroma_min', 'chroma_max', 'chroma_median', 'chroma_std', 'chroma_var', 'spectral_mean', 'spectral_min', 'spectral_max', 'spectral_median', 'spectral_std', 'spectral_var', 'tonnetz_mean', 'tonnetz_min', 'tonnetz_max', 'tonnetz_median', 'tonnetz_std', 'tonnetz_var', 'melspectrogram_mean', 'melspectrogram_min', 'melspectrogram_max', 'melspectrogram_median', 'melspectrogram_std', 'melspectrogram_var']
    if len(dataframe) == 0:
        print("DataFrame is empty. No data to classify.")
        return None

    # Only get last row of data
    data = dataframe.iloc[-1][features]
    
    # Reshape the features
    data = data.values.reshape(1, -1)
    
    # debug information
    # print("Input data shape:", data.shape)
    # print("Input data:", data)
    
    # Make prediction
    classification = model.predict(data)
    
    print("Prediction:", classification)
    
    return classification

def speak():
    message = "There is no need to bark it will be ok, I the Robot is here!"
    engine = pyttsx3.init()

    engine.setProperty('rate', 100)
    engine.setProperty('volume', 1)

    engine.say(message)
    engine.runAndWait()

def delete_audio(file_path):
    os.remove(file_path)
    print("Audio file deleted:", file_path)

def received_stop_command():
    pass

def main():
    metadata=pd.DataFrame()
    loaded_model = joblib.load('random_forest_model.pkl')
    
    print('model loaded')
    
    volume_norm = 0
    bark_counter = 0
    def callback(indata, frames, time, status):
        nonlocal volume_norm
        nonlocal bark_counter
        volume_norm = np.linalg.norm(indata) * 10
        print("|" * int(volume_norm))
        
        if volume_norm > threshold:
            print("Significant noise detected!")
            
            recorded_frames = record_audio(record_duration, sample_rate, chunk_size)
            audio_file = save_audio(recorded_frames, sample_rate)
            preprocess_audio(audio_file, metadata=metadata)
            print("Preprocessing is finished the results are:")
            print(metadata.shape)
            display(metadata)
            audio_noise = classify_audio(loaded_model, metadata)
            print(type(audio_noise))
            print(f'This noise is: {audio_noise}')
            if 'dog_bark' in np.ndarray.tolist(audio_noise):
                print('barking detected')
                bark_counter += 1
                label_bark_counter.config(text=f"Dog has barked: {bark_counter} times")
                speak()
            else:
                print('No barking detected')
                # speak()
            delete_audio(audio_file)
    
    duration = 10000
    threshold = 1
    record_duration = 3
    sample_rate = 44100
    chunk_size = 1024

    def start_audio_processing():
        detect_noise(duration, threshold, callback)

    audio_thread = threading.Thread(target=start_audio_processing)
    audio_thread.daemon = True
    audio_thread.start()

    # Tkinter GUI
    root = tk.Tk()
    root.title("Dog Companion")

    label = tk.Label(root, text="Dog Companion")
    label.pack()

    image_pil = Image.open(r"C:\Users\jsmit\.vscode\Projects\Senior Project\Image-1.jpg")
    image_pil = image_pil.resize((400, 300))
    image_tk = ImageTk.PhotoImage(image_pil)
    label = tk.Label(root, image=image_tk)
    label.pack()

    bark_counter_str = tk.StringVar()
    bark_counter_str.set(bark_counter)
    label_bark_counter = tk.Label(root, text=f"Dog has barked: {bark_counter_str.get()} times")
    label_bark_counter.pack()

    button_stop = tk.Button(root, text="Stop", command=root.destroy)
    button_stop.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
