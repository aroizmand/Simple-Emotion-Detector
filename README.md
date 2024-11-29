# Simple Emotion Detector

A Python-based emotion detection system using Mediapipe for facial landmark extraction and Random Forest for classification. It identifies simple emotions (Happy, Sad, Surprised) in real-time through webcam input, with a modular pipeline for preprocessing, training, and inference.

## How to Use

The dataset is too large to include in the repository. Download it from [Google Drive](https://drive.google.com/file/d/1xsOHIKD9rWpzvM-v4ZnByxbZCr0KiUbd/view?usp=sharing).

### From scratch:

1.  Delete files:
    `model.pkl`, `data.txt`, and `scaler.pkl`

2.  Create your venv, activate it, and install requirements:

    - `python -m venv venv`
    - `venv/Scripts/activate`
    - `pip install -r requirements.txt`

3.  Preprocess your data:

    - `python prepare_data.py`

4.  Train the model:

    - `python train_model.py`

5.  Run real-time emotion detection:
    - `python test_model.py`

### With model available:

1.  Run real-time emotion detection:
    - `python test_model.py`

## Notes

- Ensure your training data is organized in folders (happy, sad, surprised) under `./data/train/`.
- Press `q` to quit the real-time detection.
- In the file test_model make sure to select the webcam index `(default = 0)` that you want to use in case you have more than one. The line to edit is `cv2.VideoCapture(0)`

# Simple Emotion Detector

A Python-based emotion detection system using Mediapipe for facial landmark extraction and Random Forest for classification. It identifies simple emotions (Happy, Sad, Surprised) in real-time through webcam input, with a modular pipeline for preprocessing, training, and inference.

## How to Use

The dataset is too large to include in the repository. Download it from [Google Drive](https://drive.google.com/file/d/1xsOHIKD9rWpzvM-v4ZnByxbZCr0KiUbd/view?usp=sharing).

### From scratch:

1.  Delete files:
    `model.pkl`, `data.txt`, and `scaler.pkl`

2.  Create your venv, activate it, and install requirements:
    - `python -m venv venv`
    - `venv/Scripts/activate`
    - `pip install -r requirements.txt`  

3.  Preprocess your data:

    - `python prepare_data.py`

4.  Train the model:

    - `python train_model.py`

5.  Run real-time emotion detection:
    - `python test_model.py`

### With model available:

1.  Run real-time emotion detection:
    - `python test_model.py`

## Notes

- Ensure your training data is organized in folders (happy, sad, surprised) under `./data/train/`.
- Press `q` to quit the real-time detection.
- In the file test_model make sure to select the webcam index `(default = 0)` that you want to use in case you have more than one. The line to edit is `cv2.VideoCapture(0)`
