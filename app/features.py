import numpy as np

# MediaPipe eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(eye_points):
    """
    Computes Eye Aspect Ratio (EAR)
    """
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])

    return (A + B) / (2.0 * C + 1e-6)


def extract_ear(landmarks, width, height):
    """
    Extracts average EAR from both eyes
    """

    def get_points(indices):
        return np.array([
            [landmarks[i].x * width, landmarks[i].y * height]
            for i in indices
        ])

    left_eye = get_points(LEFT_EYE)
    right_eye = get_points(RIGHT_EYE)

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    return (left_ear + right_ear) / 2.0
