# Simple Emotion Detector
A Python-based emotion detection system using Mediapipe for facial landmark extraction and Random Forest for classification. It identifies emotions (Happy, Sad, Surprised) in real-time through webcam input, with a modular pipeline for preprocessing, training, and inference.

## How to Use
1. Preprocess your data:
   python prepare_data.py

2. Train the model:
  python train_model.py


3. Run real-time emotion detection:
  python test_model.py


## Notes
- Ensure your training data is organized in folders (e.g., HAPPY, SAD, SURPRISED) under `./data/train/`.
- Press `q` to quit the real-time detection.
- In the file test_model make sure to select the webcam index that you want to use in case you have more than one. The line to edit is `cv2.VideoCapture(0)`


