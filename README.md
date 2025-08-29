# 🌱 Deforestation — Fire Classification (MODIS + ML + Streamlit)

[Screencast from 2025-08-29 19-50-52.webm](https://github.com/user-attachments/assets/9e50413c-6e44-4c5a-af43-4b89775f651e)


[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end machine learning project that classifies fire types in India using **MODIS satellite fire detection data**.  
> Built with **Python, scikit-learn**, and deployed as an **interactive Streamlit web app** for real-time predictions.
> An interactive Streamlit web application that uses MODIS satellite data and a machine learning model to classify fire incidents in India into Vegetation Fire, Other Static Land Source, or Offshore Fire. The app enables users to enter satellite readings (e.g., brightness, FRP, confidence) and get real-time predictions, supporting faster and more accurate disaster management decisions.

---

## ✨ What this project does
- Ingests **MODIS satellite fire detections**.
- Cleans & prepares data (missing values, encodings, scaling).
- Engineers/uses key features: `brightness`, `brightness_t31`, `frp`, `scan`, `track`, `confidence`.
- Trains ML models and selects the best via **Accuracy** & **F1-score**.
- Deploys a **Streamlit** app for instant classification:
  - **Vegetation Fire**
  - **Other Static Land Source**
  - **Offshore Fire**

---

## 🚀 Quick Start

### 1) Clone the repo
```bash
git clone https://github.com/tejth/Deforestation.git
cd <your-repo>
