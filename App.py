# App.py
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import plotly.graph_objects as go
from collections import deque
import os

# ──────────────────────────────────────────────────────
# Suppress TensorFlow noise
# ──────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ──────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────
st.set_page_config(page_title="Drowsiness Detector", page_icon="car", layout="centered")

st.title("Driver Drowsiness Detection System")
st.markdown("**Real-time 3-class detection** | **Alarm if eyes closed > 2 sec**")

# ──────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "uta_rldd_lstm.h5"
    if not os.path.exists(model_path):
        st.error(f"Model `{model_path}` not found in project root.")
        st.stop()
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

# ──────────────────────────────────────────────────────
# MediaPipe
# ──────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

LEFT_EYE  = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]

def eye_aspect_ratio(landmarks):
    v1 = np.linalg.norm(landmarks[1] - landmarks[5])
    v2 = np.linalg.norm(landmarks[2] - landmarks[4])
    h  = np.linalg.norm(landmarks[0] - landmarks[3])
    return (v1 + v2) / (2 * h + 1e-6)

# ──────────────────────────────────────────────────────
# Sidebar Controls (ONLY ONE CHECKBOX!)
# ──────────────────────────────────────────────────────
st.sidebar.header("Controls")

# This is the **only** checkbox with this key
run = st.sidebar.checkbox("Start Webcam", value=False, key="start_webcam")

ear_thresh = st.sidebar.slider("EAR Threshold", 0.10, 0.30, 0.20, 0.01, key="ear")
alarm_sec = st.sidebar.slider("Alarm Trigger (sec)", 1.0, 5.0, 2.0, 0.5, key="alarm")

# ──────────────────────────────────────────────────────
# UI Placeholders
# ──────────────────────────────────────────────────────
frame_ph = st.empty()
ear_chart = st.empty()
status_ph = st.empty()
alarm_ph = st.empty()

CLASS_NAMES = ["Alert", "Low Vigilant", "Drowsy"]

# ──────────────────────────────────────────────────────
# Main Detection Loop
# ──────────────────────────────────────────────────────
if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not accessible. Check permissions.")
        st.stop()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    closed_frames = 0
    max_closed = int(alarm_sec * 30)
    start_time = time.time()

    if "seq" not in st.session_state:
        st.session_state.seq = []

    ear_hist = deque(maxlen=120)
    time_hist = deque(maxlen=120)

    while run:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear_val = None
        pred_text = "No face"
        confidence = 0.0

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            landmarks = np.array([(lm.landmark[i].x, lm.landmark[i].y, lm.landmark[i].z) 
                                  for i in range(468)])

            # EAR
            left_ear = eye_aspect_ratio(landmarks[LEFT_EYE])
            right_ear = eye_aspect_ratio(landmarks[RIGHT_EYE])
            ear_val = (left_ear + right_ear) / 2.0

            # Sequence
            st.session_state.seq.append(landmarks.flatten())
            if len(st.session_state.seq) > 30:
                st.session_state.seq.pop(0)

            # Predict
            if len(st.session_state.seq) == 30:
                seq = np.expand_dims(st.session_state.seq, axis=0)
                pred = model.predict(seq, verbose=0)[0]
                idx = np.argmax(pred)
                confidence = pred[idx]
                pred_text = f"{CLASS_NAMES[idx]} ({confidence:.2f})"

            # Alarm
            if ear_val < ear_thresh:
                closed_frames += 1
            else:
                closed_frames = 0

            if closed_frames > max_closed:
                cv2.putText(frame, "ALARM!", (40, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 5)
                alarm_ph.error("DROWSY DETECTED!")

            cv2.putText(frame, f"EAR: {ear_val:.3f}", (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cur_time = time.time() - start_time
            ear_hist.append(ear_val)
            time_hist.append(cur_time)

        else:
            st.session_state.seq = []

        # Update UI
        frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        status_ph.markdown(f"**Status:** {pred_text}")

        if len(ear_hist) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(time_hist), y=list(ear_hist),
                                     mode='lines', name='EAR', line=dict(color='cyan')))
            fig.add_hline(y=ear_thresh, line_dash="dash", line_color="red")
            fig.update_layout(height=260, margin=dict(l=0,r=0,t=30,b=0),
                              title="EAR Live", xaxis_title="Time (s)", yaxis_title="EAR")
            ear_chart.plotly_chart(fig, use_container_width=True)

        # **NO SECOND CHECKBOX** — read current state from session
        run = st.session_state.start_webcam

        time.sleep(0.03)

    cap.release()
    frame_ph.empty()
    ear_chart.empty()
    status_ph.empty()
    alarm_ph.empty()

else:
    st.info("Check **Start Webcam** in the sidebar to begin.")
    st.markdown("**Tip:** Allow camera → keep face in frame → good lighting.")