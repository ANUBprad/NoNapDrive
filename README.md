# 🚗 NoNapDrive
### Real-Time Driver Drowsiness Detection using AI

NoNapDrive is an intelligent, real-time driver monitoring system that detects **drowsiness and microsleep** using computer vision and deep learning. It analyzes eye behavior through a webcam and triggers **severity-based alerts** to help prevent fatigue-related accidents.

---

## ✨ Highlights

- 🎥 Live webcam-based detection  
- 👁️ Eye Aspect Ratio (EAR) using MediaPipe Face Mesh  
- 🧠 3-Class LSTM model  
  - 🟢 Alert  
  - 🟡 Drowsy  
  - 🔴 Critical (Microsleep)  
- ⏱️ Temporal validation to reduce false alarms  
- 🔊 Escalating audio alerts  
- ⚙️ Fully configurable via YAML  
- 💻 Software-only (no hardware required)

---

## 🧩 How It Works

1. Webcam captures real-time video  
2. Face landmarks detected using MediaPipe  
3. Eye Aspect Ratio (EAR) is computed  
4. EAR sequences are fed to an LSTM model  
5. Driver state is classified  
6. Alert is triggered based on severity  

---

## 🛠️ Tech Stack

| Component | Technology |
|--------|-----------|
| Language | Python 3.10 |
| Deep Learning | TensorFlow (tf.keras) |
| Vision | MediaPipe, OpenCV |
| UI | Streamlit |
| Utilities | NumPy, YAML |

---

## ▶️ Getting Started

### 1️⃣ Create virtual environment (Python 3.10)
```
py -3.10 -m venv venv310
venv310\Scripts\activate
```

2️⃣ Install dependencies
```
pip install -r requirements.txt
```

3️⃣ Train the model (one-time)
```
jupyter notebook
```
Run:
```
notebooks/drowsiness_model_3class.ipynb
```
Move the trained model to:
```
models/drowsiness_lstm_3class_tf.keras
```

4️⃣ Run the application
```
streamlit run app/app.py
```
Open in browser:
```
http://localhost:8501
```

## 🚨 Alert Logic

- Alert	-> Normal monitoring
- Drowsy	->	Mild warning beep
- Critical	->	Loud alarm after time confirmation

## ⚙️ Configuration
All thresholds and parameters are configurable via config.yaml, allowing easy tuning without modifying code.

## 👤 Author

Anubhab Pradhan
BE – Artificial Intelligence & Data Science
CMR Institute of Technology, Bangalore

## 📜 License

For academic and educational use only.
