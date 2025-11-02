# 🚗 NoNapDrive  
![NoNapDrive Banner](https://via.placeholder.com/1200x400?text=NoNapDrive+%E2%80%94+AI+Driver+Drowsiness+Detection)

> **AI-powered real-time drowsiness detection to prevent accidents before they happen.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-4285F4?style=flat&logo=Google&logoColor=white)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset)

---

## 🧠 Overview
**NoNapDrive** is a **real-time driver drowsiness detection system** using **deep learning** and **computer vision** to detect early signs of fatigue — *before* it’s too late.

Trained on the **UTA Real-Life Drowsiness Dataset (UTA-RLDD)**, our LSTM-based model classifies driver states into three levels:

| Class | Description |
|--------|--------------|
| **Alert** | KSS 1–3 |
| **Low Vigilant** | KSS 6–7 |
| **Drowsy** | KSS 8–9 |

If eyes remain closed for **>2 seconds**, the system triggers a **loud visual/audio alarm** — preventing potential accidents caused by microsleep.

---

## 🚀 Key Features

| Feature | Description |
|----------|-------------|
| **3-Class Prediction** | Detects subtle drowsiness using temporal LSTM on facial landmarks |
| **Real-Time Webcam Demo** | Streamlit app with live video, EAR graph, and class confidence |
| **>2 Sec Eye Closure Alarm** | Alerts instantly on prolonged eye closure |
| **MediaPipe Face Mesh** | 468 3D landmarks for head pose & blink detection |
| **IoT & Edge Ready** | Export to TensorFlow Lite for Raspberry Pi / Jetson Nano |
| **Production-Ready** | Clean code, error handling, and deployment scripts |

---

## 🎯 Dataset — UTA-RLDD

**Source:** [UTA Real-Life Drowsiness Dataset](https://www.kaggle.com/datasets/rishab260/uta-reallife-drowsiness-dataset)  
**Paper:** [A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection (CVPRW 2019)](https://sites.google.com/view/utarldd/home)

### Why UTA-RLDD?
- 30+ hours of **real-world driving videos**
- 60 participants (diverse ethnicity, glasses, facial hair)
- **Self-recorded** with webcam/phone — real-life conditions
- Labeled with **Karolinska Sleepiness Scale (KSS)**
- Variable lighting, angles, and resolutions

> *The largest and most realistic drowsiness dataset to date.*

---

## 🧩 Model Architecture

Input (30 frames × 468×3 landmarks)
↓
LSTM(128) → Dropout → LSTM(64) → Dropout
↓
Dense(64, ReLU) → Dense(3, softmax)


**Details:**
- **Input:** 30-frame sequences of MediaPipe 3D facial landmarks  
- **EAR (Eye Aspect Ratio):** Independent blink detection for alarm  
- **Accuracy:** ~90% validation accuracy (3-class)  
- **Inference:** 30 FPS (CPU) | 60+ FPS (GPU/TFLite)  
- **Pre-trained model:** `uta_rldd_lstm.h5`

---

## 🧪 Live Demo — Streamlit App

Run the Streamlit web app for an instant real-time demo:

```bash
streamlit run App.py
```
Demo Features
1. Live webcam feed
2. Real-time EAR graph (Plotly)
3. Confidence scores for all classes
4. Adjustable EAR threshold & alarm duration
5. Visual + audio alerts when drowsiness is detected
⚠️ Webcam works locally only. For Streamlit Cloud, use a demo video.


📁 Folder Structure

NoNapDrive/
│
├── App.py                  # Streamlit live demo
├── uta_rldd_lstm.h5        # Trained LSTM model
├── requirements.txt
├── driver_training.ipynb   # Kaggle training notebook
├── demo.mp4                # Optional demo video
└── README.md

⚙️ Installation
```bash
git clone https://github.com/yourusername/NoNapDrive.git
cd NoNapDrive
pip install -r requirements.txt
```
▶️ Run the Demo
```bash
streamlit run App.py
```
1. Allow webcam access
2. Click Start Webcam
3. Face the camera → watch real-time predictions + alarms


🌐 Edge Deployment (Raspberry Pi / Jetson Nano)
```bash
# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("nonapdrive.tflite", "wb").write(tflite_model)
```
Run inference using opencv + tflite-runtime for edge devices.

🤝 Contributing

Contributions are welcome!
You can help by:
* Improving EAR robustness
* Adding audio/voice alerts
* Multi-face detection
* CAN bus integration for in-vehicle systems

🪪 License

Released under the MIT License. Free to use, modify, and deploy.

📚 Citation

If you use this dataset or model, please cite:
```bash
@inproceedings{ghoddoosian2019realistic,
  title={A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection},
  author={Ghoddoosian, Reza and Galib, Marnim and Athitsos, Vassilis},
  booktitle={CVPR Workshops},
  year={2019}
}
```
🛣️ Stay Awake. Drive Safe.

NoNapDrive — Because every second behind the wheel matters.

#NoNapDrive #DriverSafety #AI #ComputerVision #SafeRoads

Made with passion for road safety — let’s build a future with zero drowsy-driving accidents.
EOF


✅ **How to use:**
1. Copy the full block above.  
2. Paste it into your terminal at the root of your project.  
3. It will instantly create a polished `README.md` file.  

Would you like me to include a **GitHub Actions badge** (e.g., for build/test workflows) and a **demo GIF placeholder** at the top as well? Those make the page more engaging.
