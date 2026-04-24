---
title: Unveiling Climate Change Dynamics Through Earth Surface Temperature Analysis
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# 🌍 Unveiling Climate Change Dynamics Through Earth Surface Temperature Analysis

> A production-grade Flask ML web app for exploring Earth's surface temperature records, visualising climate change trends, and predicting future temperatures using trained ML models.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)

---

## ✨ Features

- **Interactive Charts** — Annual trends, decadal averages, and seasonal cycles via Chart.js
- **Dual ML Prediction** — City-level (local temperature) and Global Average (climate science metric)
- **Live Weather** — Real-time conditions from Open-Meteo API (free, no key needed)
- **20+ Cities** — Hyderabad, Mumbai, Delhi, Tokyo, London, New York, and more
- **Dark Glassmorphism UI** — Animated gradient background, responsive on all devices
- **Forecast to 2100** — Linear Regression + Random Forest trained on Berkeley Earth data

---

## 🚀 Run Locally

```bash
# 1. Clone / unzip the project
cd climate-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python app.py
```

Open: **http://127.0.0.1:7860**

---

## 📁 Project Structure

```
climate-dashboard/
├── app.py                    # Flask application (all routes + ML logic)
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python version for Hugging Face
├── README.md                 # This file
├── templates/
│   ├── base.html             # Common layout (navbar, footer, canvas)
│   ├── index.html            # Landing page
│   ├── analysis.html         # EDA charts + model metrics
│   ├── predict.html          # City + Global ML prediction
│   ├── about.html            # Project info + tech stack
│   └── contact.html          # Contact form
├── static/
│   ├── css/style.css         # Production CSS (dark theme, glassmorphism)
│   └── js/main.js            # Animated background + UX interactions
├── model/
│   ├── best_model.pkl        # Best city model
│   ├── lr_model.pkl          # City Linear Regression
│   ├── rf_model.pkl          # City Random Forest
│   ├── global_best_model.pkl # Best global model
│   ├── global_lr_model.pkl   # Global Linear Regression
│   ├── global_rf_model.pkl   # Global Random Forest
│   ├── cities.pkl            # City metadata + monthly averages
│   ├── metrics.pkl           # City model evaluation metrics
│   └── global_metrics.pkl    # Global model evaluation metrics
└── dataset/
    ├── GlobalTemperatures.csv
    └── CityTemperatures.csv
```

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| GET | `/analysis` | Charts + EDA |
| GET/POST | `/predict` | ML prediction UI |
| GET | `/about` | About the project |
| GET | `/contact` | Contact form |
| POST | `/api/predict` | City temp prediction (JSON) |
| POST | `/api/global` | Global temp prediction (JSON) |
| GET | `/api/weather?city=X` | Live weather for city |
| GET | `/api/cities` | List all supported cities |

---

## 🌐 Deploy on Hugging Face Spaces

This app is ready for Hugging Face Spaces with Docker SDK.

The `README.md` header (at top of this file) contains the required metadata.

App runs on **port 7860** via:
```python
port = int(os.environ.get("PORT", 7860))
app.run(host="0.0.0.0", port=port)
```

---

## 📊 Dataset

- **Berkeley Earth Surface Temperature** dataset
- **GlobalTemperatures.csv** — 1750–2015 monthly global land averages
- **CityTemperatures.csv** — City-level historical temperatures

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, Flask 3.0 |
| ML | scikit-learn (LinearRegression, RandomForestRegressor) |
| Data | pandas, numpy |
| Charts | Chart.js 4 |
| Weather API | Open-Meteo (free, no key) |
| Fonts | Syne, Manrope, DM Mono (Google Fonts) |
| Hosting | Hugging Face Spaces (Docker) |

---

*Built with 🌍 to make climate data accessible.*
