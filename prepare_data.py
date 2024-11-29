import os
import cv2
import numpy as np
from utils import get_face_landmarks

DATA_DIR = './data/train'
OUTPUT_FILE = 'data.txt'

def preprocess_and_save_data(data_dir, output_file):
    output = []
    for emotion_idx, emotion in enumerate(sorted(os.listdir(data_dir))):
        emotion_dir = os.path.join(data_dir, emotion)
        for image_file in os.listdir(emotion_dir):
            image_path = os.path.join(emotion_dir, image_file)
            img = cv2.imread(image_path)
            face_landmarks = get_face_landmarks(img)

            if face_landmarks and len(face_landmarks) == 1404:  # Valid landmarks
                face_landmarks.append(emotion_idx)  # Add label
                output.append(face_landmarks)

    np.savetxt(output_file, np.asarray(output), delimiter=',')
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_and_save_data(DATA_DIR, OUTPUT_FILE)
