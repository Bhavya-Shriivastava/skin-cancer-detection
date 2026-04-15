# 🧠 Skin Cancer Detection using CNN + Hybrid Model

## 🚀 Overview

This project is a **Skin Cancer Detection System** that combines:

* 🧠 Deep Learning (CNN trained from scratch)
* 📊 Patient metadata (age, sex, symptoms, family history)

to provide a **risk-based prediction** of whether a skin lesion is cancerous or not.

---

## 🎯 Key Features

* 🔍 Image-based cancer detection using CNN
* 🧾 Hybrid risk scoring with patient information
* 🌐 Interactive frontend (HTML, CSS, JS)
* ⚡ Backend API using Flask
* 📊 Confidence score + risk level output

---

## 🏗️ Project Structure

```
skin-cancer-detection/
│
├── backend/
│   ├── app.py
│   ├── train_cnn_scratch.py
│   ├── evaluate_binary_model.py
│   ├── predict_hybrid.py
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
└── README.md
```

---

## 🧠 Model Details

* Architecture: Custom CNN (built from scratch)
* Input Size: 224 × 224 × 3
* Output: Binary (Cancer / Non-Cancer)
* Activation: Sigmoid
* Loss Function: Binary Crossentropy

---

## ⚙️ Tech Stack

* Python 🐍
* TensorFlow / Keras
* Flask (Backend API)
* HTML, CSS, JavaScript (Frontend)

---

## 📊 How It Works

1. User uploads a skin image
2. CNN model predicts cancer probability
3. Patient data is incorporated:

   * Age
   * Sex
   * Family history
   * Symptoms
4. Final **risk score** is computed
5. Output:

   * Prediction (Benign / Malignant / Uncertain)
   * Confidence %
   * Risk Level

---

## ▶️ How to Run

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/skin-cancer-detection.git
cd skin-cancer-detection
```

### 2. Install dependencies

```
pip install -r backend/requirements.txt
```

### 3. Run backend

```
cd backend
python app.py
```

### 4. Open frontend

* Open `frontend/index.html` in browser

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.
It is **not a medical diagnostic tool**.

---

## 👨‍💻 Author

**Bhavya Shrivastava**

* B.Tech CSE, KIIT University
* Aspiring Software Engineer
* GitHub: https://github.com/Bhavya-Shriivastava
