import cv2
import pickle
from utils import get_face_landmarks

MODEL_FILE = 'model.pkl'
SCALER_FILE = 'scaler.pkl'
EMOTIONS = ['HAPPY', 'SAD', 'SURPRISED']

def main():
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_landmarks = get_face_landmarks(frame, static_image_mode=False)

        if face_landmarks and len(face_landmarks) == 1404:
            scaled_landmarks = scaler.transform([face_landmarks])
            emotion_idx = model.predict(scaled_landmarks)[0]
            emotion = EMOTIONS[int(emotion_idx)]
            cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
