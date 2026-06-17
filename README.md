# 🩺 Multimodal Kidney Disease Assessment Using Clinical Data and Renal Ultrasound Imaging

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Web_App-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)
![XGBoost](https://img.shields.io/badge/XGBoost-Clinical_Model-orange)
![License](https://img.shields.io/badge/License-MIT-success)

An AI-powered healthcare application that predicts **Chronic Kidney Disease (CKD)** using **Clinical Parameters**, **Renal Ultrasound Images**, and a **Multimodal Fusion Model**.

</div>

---

# 📸 Application Preview

## 🏠 Dashboard

![Dashboard](assets/dashboard.png)

The dashboard provides access to:

- Clinical Assessment
- Ultrasound Analysis
- Patient Reports
- Project Overview

---

## 📊 Clinical Assessment

![Clinical Assessment](assets/clinical.png)

The Clinical Assessment page allows users to enter patient information including:

- Age
- Blood Pressure
- Hemoglobin
- Albumin
- Serum Creatinine
- Diabetes
- Hypertension

The XGBoost model predicts whether the patient is likely to have CKD.

---

## 🩻 Renal Ultrasound Analysis
![Ultrasound Analysis](assets/ultrasound.png)

Upload a renal ultrasound image.

The ResNet18 deep learning model analyzes the kidney ultrasound and predicts:

- CKD
- Normal

---

## 🧠 Multimodal Prediction

![Prediction](assets/result.png)

The application combines:

- Clinical Prediction
- Ultrasound Prediction

to produce the final diagnosis.

Displayed results include:

- Clinical Prediction
- Ultrasound Prediction
- Fusion Prediction
- Confidence Score

---

## 📄 Medical Report

![Prediction](assets/result.png)

Automatically generated report includes:

- Patient Details
- Clinical Result
- Ultrasound Result
- Final Fusion Prediction
- AI Summary

Reports can be downloaded as PDF.

---

# 🏗️ System Architecture

![Architecture](assets/architecture.png)

---

# 🔄 Workflow

![Workflow](assets/workflow.png)

---

# 🚀 Features

✅ Clinical CKD Prediction

✅ Ultrasound Image Analysis

✅ Multimodal Fusion Model

✅ Flask Web Application

✅ PDF Report Generation

✅ Responsive Dashboard

✅ Medical-Themed User Interface

---

# 🧠 Machine Learning Models

## Clinical Model

Algorithm:

- XGBoost

Input Features

- Age
- Blood Pressure
- Albumin
- Serum Creatinine
- Hemoglobin
- Diabetes
- Hypertension
- Specific Gravity
- Additional Clinical Parameters

Output

- CKD
- Normal

---

## Ultrasound Model

Architecture

ResNet18

Framework

PyTorch

Input

Renal Ultrasound Image

Output

- CKD
- Normal

---

## Fusion Model

Combines:

Clinical Prediction

+

Ultrasound Prediction

↓

Final Diagnosis

---

# 📂 Project Structure

```text
Multimodal_Kidney_Project
│
├── app.py
├── requirements.txt
├── runtime.txt
├── render.yaml
│
├── models
│   ├── ckd_xgb_model.pkl
│   ├── ultrasound_model.pth
│   ├── feature_columns.pkl
│   ├── label_encoders.pkl
│   └── class_names.pkl
│
├── static
│   ├── css
│   ├── images
│   └── js
│
├── templates
│   ├── dashboard_v2.html
│   ├── numerical_v2.html
│   ├── ultrasound_v2.html
│   ├── result_v2.html
│   └── report.html
│
└── assets
```

---

# 🛠️ Technologies Used

### Backend

- Flask

### Deep Learning

- PyTorch

### Machine Learning

- XGBoost
- Scikit-Learn

### Frontend

- HTML
- CSS
- JavaScript

### Other Libraries

- Pandas
- NumPy
- Pillow
- ReportLab

---

# ⚙️ Installation

```bash
git clone https://github.com/suchindra654-ctrl/Multimodal_Kidney_Project.git

cd Multimodal_Kidney_Project

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

python app.py
```

Open

```
http://127.0.0.1:5000
```

---

# 📈 Future Improvements

- Explainable AI (Grad-CAM)
- Vision Transformers
- Doctor Portal
- Patient Login
- CKD Stage Prediction
- Cloud Database
- Mobile Application

---

# 👨‍💻 Author

**Suchindra**

Machine Learning & AI Engineering Student

GitHub:

https://github.com/suchindra654-ctrl

---

⭐ If you found this project useful, consider giving it a star!