import cv2
import mediapipe as mp

def get_face_landmarks(image, static_image_mode=True):
    """
    Extract face landmarks using Mediapipe FaceMesh.
    Normalizes landmarks to the range [0, 1] relative to the face bounding box.

    :param image: Input image (numpy array).
    :param static_image_mode: Whether the input image is static.
    :return: List of normalized landmarks or None if no landmarks are found.
    """
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    results = face_mesh.process(image_input_rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    xs, ys, zs = zip(*[(lm.x, lm.y, lm.z) for lm in landmarks])
    min_x, min_y, min_z = min(xs), min(ys), min(zs)
    max_x, max_y, max_z = max(xs), max(ys), max(zs)

    normalized_landmarks = []
    for x, y, z in zip(xs, ys, zs):
        normalized_landmarks.extend([
            (x - min_x) / (max_x - min_x),
            (y - min_y) / (max_y - min_y),
            (z - min_z) / (max_z - min_z)
        ])

    return normalized_landmarks
