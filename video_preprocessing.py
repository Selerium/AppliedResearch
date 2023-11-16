# processing videos frame by frame

import cv2
import numpy as np
import os

def load_and_preprocess_video(video_path, target_height, target_width, frames_per_clip):
    frames = []

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (target_width, target_height))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    if len(frames) < frames_per_clip:
        return None
    selected_frames = frames[:frames_per_clip]
    return np.array(selected_frames)

video_directory = ""
video_data = []
labels = []
target_height = 64
target_width = 64
frames_per_clip = 30

for video_file in os.listdir(video_directory):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_directory, video_file)
        video_frames = load_and_preprocess_video(video_path, target_height, target_width, frames_per_clip)

        if video_frames is not None:
            video_data.append(video_frames)

            label = video_file.split("_")[0]
            labels.append(label)

video_data = np.array(video_data)
labels = np.array(labels)
print("Video Data Shape:", video_data.shape)
print("Labels Shape:", labels.shape)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import to_categorical

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
one_hot_labels = to_categorical(encoded_labels)
X_train, X_test, y_train, y_test = train_test_split(video_data, one_hot_labels, test_size=0.2, random_state=42)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)