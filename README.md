# 🔧 Bearing Fault Detector
### Real-Time Industrial Vibration Signal Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?style=flat-square&logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-97.89%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## 🌐 Live Demo
**[👉 Open Live App](https://bearing-fault-detector.onrender.com)**

---

## 📌 What Is This?

This project implements the core signal processing and machine learning pipeline used in industrial predictive maintenance systems — built from scratch as a fully functional web application.

Bearings are critical components in motors, turbines, and industrial machinery. A failed bearing can cause unplanned downtime costing lakhs of rupees per hour. This system detects faults early by analyzing vibration signals — before physical failure occurs.

---

## 🎯 Fault Classes Detected

| Condition | Description |
|-----------|-------------|
| ✅ Normal | Healthy bearing, no anomaly detected |
| 🟠 Ball Fault | Defect on the rolling ball element |
| 🔴 Inner Race Fault | Defect on the inner raceway |
| 🟣 Outer Race Fault | Defect on the outer raceway |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Classifier |
| Dataset | CWRU Bearing Benchmark |
| Accuracy | **97.89%** |
| Validation | 5-Fold Cross Validation |
| Noise Robustness | Tested with Gaussian noise injection |

---

## ⚙️ How It Works
BearingFaultdetector/
├── app.py                  # Flask backend — prediction + SSE streaming
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment configuration
├── templates/
│   └── index.html          # Interactive dashboard
├── static/                 # Static assets
├── pipeline.pkl            # Trained Random Forest pipeline
├── label_encoder.pkl       # Fault class label encoder
└── features_dataset.csv    # Extracted features dataset

---

## 📂 Dataset

Uses the **[CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter)** — the standard IEEE benchmark used in hundreds of published research papers.

- Sampling rate: 12,000 Hz
- Bearing: 6205-2RS JEM SKF deep groove ball bearing
- Fault sizes: 0.007, 0.014, 0.021 inches
- Signal type: Drive End (DE) accelerometer

---

## 💡 Real World Context

The pipeline here — vibration signal → feature extraction → ML classification — is conceptually identical to what industrial predictive maintenance systems use. This project demonstrates understanding of the full stack from raw sensor data to deployed web interface.

---

## 👩‍💻 Author

**Anjali Sharma**
ECE Final Year Student
🔗 [GitHub](https://github.com/ANJALISHARMA1600)

---

## 📄 License

MIT License — free to use, modify, and distribute.