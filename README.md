# 🚗 NoNapDrive  

### Real-Time Driver Drowsiness Detection using AI

NoNapDrive is an intelligent **real-time driver monitoring system** that detects **drowsiness and microsleep** using computer vision and deep learning.  
It analyzes eye behavior through a webcam and triggers **severity-based alerts** to help prevent fatigue-related road accidents.

---

## ✨ Key Features

- 🎥 Live webcam-based monitoring  
- 👁️ Eye Aspect Ratio (EAR) calculation using MediaPipe Face Mesh  
- 🧠 LSTM-based deep learning model (3-class classification)  
- 🟢 Alert | 🟡 Drowsy | 🔴 Critical (Microsleep)  
- ⏱️ Temporal validation to reduce false positives  
- 🔊 Escalating audio alerts based on severity  
- ⚙️ Fully configurable using YAML  
- 💻 Software-only solution (no additional hardware required)

---

## 🧩 System Workflow

1. Webcam captures real-time video frames  
2. Face landmarks are detected using MediaPipe  
3. Eye Aspect Ratio (EAR) is computed per frame  
4. Sequential EAR data is passed to an LSTM model  
5. Driver state is classified into 3 levels  
6. Corresponding alert is triggered  

---

## 🛠️ Tech Stack

| Component | Technology |
|--------|------------|
| Programming Language | Python 3.10 |
| Deep Learning | TensorFlow (tf.keras) |
| Computer Vision | MediaPipe, OpenCV |
| Web Interface | Streamlit |
| Utilities | NumPy, PyYAML |

---

## 📁 Project Structure
```
NoNapDrive/
│
├── app/
│ ├── pycache/
│ ├── alert.py # Audio alert handling logic
│ ├── app.py # Streamlit application entry point
│ ├── features.py # EAR feature extraction
│ ├── model.py # LSTM model loading & inference
│ └── state.py # Driver state management
│
├── assets/
│ ├── alarm.wav # Critical alert sound
│ └── beep.wav # Mild warning sound
│
├── models/
│ └── drowsiness_lstm_3class_tf.keras # Trained LSTM model
│
├── notebooks/
│ └── drowsiness_model.ipynb # Model training notebook
│
├── config.yaml # Thresholds & runtime configuration
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore
```

---

## ▶️ Getting Started

### 1️⃣ Create Virtual Environment (Python 3.10 recommended)

```
py -3.10 -m venv venv310
venv310\Scripts\activate
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Train the Model (One-Time)

```
jupyter notebook
```

Run:
```
notebooks/drowsiness_model.ipynb
```

After training, place the model file inside:
```
models/drowsiness_lstm_3class_tf.keras
```

### 4️⃣ Run the Application

```
streamlit run app/app.py
```

Open in browser:
```
http://localhost:8501
```

---

## 🚨 Alert Logic

🟢 refers Alert	-> Normal monitoring
🟡 refers Drowsy	-> Mild warning beep
🔴 refers Critical	-> Loud alarm after time confirmation


## ⚙️ Configuration

All thresholds and runtime parameters can be tuned using:
```
config.yaml
```
This allows behavior changes without modifying code.

---

## 📌 Use Cases

1. Driver safety systems
2. Long-distance driving assistance
3. Academic research in computer vision
4. Fatigue detection systems


## 👤 Author

Anubhab Pradhan

BE – Artificial Intelligence & Data Science

CMR Institute of Technology, Bangalore


## 📜 License

This project is intended for academic and educational use only.
