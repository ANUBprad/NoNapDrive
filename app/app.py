import streamlit as st
import cv2
import yaml
import mediapipe as mp
from collections import deque

from features import extract_ear
from model import Predictor
from state import DrowsyState
from alert import drowsy_alert, critical_alert


# Load config
config = yaml.safe_load(open("config.yaml"))

SEQUENCE_LENGTH = config["SEQUENCE_LENGTH"]
DROWSY_TIME = config["DROWSY_TIME"]

# Streamlit setup
st.set_page_config(page_title="NoNapDrive", layout="wide")
st.title("🚗 NoNapDrive – Driver Drowsiness Detection")

run = st.toggle("Start Detection")

frame_box = st.empty()
status_box = st.empty()
ear_box = st.empty()

# OpenCV + MediaPipe
cap = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.7
)

# Model & state
predictor = Predictor(
    "models/drowsiness_lstm_3class_tf.keras",
    SEQUENCE_LENGTH
)
state = DrowsyState(DROWSY_TIME)

ear_smoother = deque(maxlen=5)

while run:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        landmarks = results.multi_face_landmarks[0].landmark

        ear = extract_ear(landmarks, w, h)
        ear_smoother.append(ear)
        ear_avg = sum(ear_smoother) / len(ear_smoother)

        ear_box.metric("EAR", round(ear_avg, 3))

        label = predictor.update(ear_avg)

        if label == "ALERT":
            status_box.success("🟢 ALERT")
            state.reset()

        elif label == "DROWSY":
            status_box.warning("🟡 DROWSY – Stay Alert")
            drowsy_alert()

        elif label == "CRITICAL":
            if state.confirm():
                status_box.error("🔴 CRITICAL – TAKE A BREAK NOW")
                critical_alert()

    frame_box.image(rgb)

cap.release()
