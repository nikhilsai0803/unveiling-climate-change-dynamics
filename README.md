# 🌍 Climate Change Dynamics — Earth Surface Temperature Analysis

> A production-grade Flask ML web app for exploring Earth's surface temperature records, visualising climate change trends, and predicting future temperatures using trained ML models.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/sharmasai12/Unveiling-climate-change-dynamics-through-earth-surface-temperature-analysis)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

---

## 🔴 Live Demo

**👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/sharmasai12/Unveiling-climate-change-dynamics-through-earth-surface-temperature-analysis)**

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-xl-dark.svg)](https://huggingface.co/spaces/sharmasai12/Unveiling-climate-change-dynamics-through-earth-surface-temperature-analysis)

> No sign-up required. Runs entirely in the browser via Hugging Face's hosted Docker runtime.

---

## ✨ Features

| | Feature | Description |
|---|---|---|
| 📈 | **Interactive Charts** | Annual trends, decadal averages, and seasonal cycles via Chart.js 4 |
| 🤖 | **Dual ML Prediction** | City-level and Global Average forecasts using two model types |
| 🌤 | **Live Weather** | Real-time conditions from Open-Meteo API — free, no key needed |
| 🏙 | **20+ Cities** | Hyderabad, Mumbai, Delhi, Tokyo, London, New York, and more |
| 🎨 | **Dark Glassmorphism UI** | Animated gradient background, responsive on all devices |
| 📅 | **Forecast to 2100** | Linear Regression + Random Forest trained on Berkeley Earth data |

---

## 🚀 Quick Start

```bash
# 1. Clone the project
git clone https://huggingface.co/spaces/sharmasai12/Unveiling-climate-change-dynamics-through-earth-surface-temperature-analysis
cd climate-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

---

## 📁 Project Structure

```
climate-dashboard/
│
├── app.py                        # Flask application — all routes + ML logic
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version for Hugging Face
│
├── templates/
│   ├── base.html                 # Common layout (navbar, footer, canvas)
│   ├── index.html                # Landing page
│   ├── analysis.html             # EDA charts + model metrics
│   ├── predict.html              # City + Global ML prediction UI
│   ├── about.html                # Project info + tech stack
│   └── contact.html              # Contact form
│
├── static/
│   ├── css/style.css             # Production CSS (dark theme, glassmorphism)
│   └── js/main.js                # Animated background + UX interactions
│
├── model/
│   ├── best_model.pkl            # Best city model (auto-selected)
│   ├── lr_model.pkl              # City — Linear Regression
│   ├── rf_model.pkl              # City — Random Forest
│   ├── global_best_model.pkl     # Best global model (auto-selected)
│   ├── global_lr_model.pkl       # Global — Linear Regression
│   ├── global_rf_model.pkl       # Global — Random Forest
│   ├── cities.pkl                # City metadata + monthly averages
│   ├── metrics.pkl               # City model evaluation metrics
│   └── global_metrics.pkl        # Global model evaluation metrics
│
└── dataset/
    ├── GlobalTemperatures.csv    # 1750–2015 monthly global land averages
    └── CityTemperatures.csv      # City-level historical temperatures
```

---

## 🔗 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Home / landing page |
| `GET` | `/analysis` | EDA charts + model metrics |
| `GET POST` | `/predict` | ML prediction UI |
| `GET` | `/about` | About the project |
| `GET` | `/contact` | Contact form |
| `POST` | `/api/predict` | City temperature prediction — returns JSON |
| `POST` | `/api/global` | Global temperature prediction — returns JSON |
| `GET` | `/api/weather?city=X` | Live weather for a given city |
| `GET` | `/api/cities` | List all supported cities |

<details>
<summary><strong>Example — City Prediction</strong></summary>

```bash
curl -X POST https://your-space.hf.space/api/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "Hyderabad", "year": 2075, "month": 6, "model_type": "best"}'
```

```json
{
  "temperature": 32.47,
  "model": "Random Forest",
  "city": "Hyderabad",
  "year": 2075
}
```

</details>

<details>
<summary><strong>Example — Global Prediction</strong></summary>

```bash
curl -X POST https://your-space.hf.space/api/global \
  -H "Content-Type: application/json" \
  -d '{"year": 2100, "model_type": "best"}'
```

```json
{
  "temperature": 10.84,
  "model": "Linear Regression",
  "year": 2100
}
```

</details>

---

## 🧠 ML Models

Two model types are trained and evaluated for both city-level and global predictions:

| Model | Strengths | Use Case |
|-------|-----------|----------|
| **Linear Regression** | Fast, interpretable, good for long-range trends | Baseline + extrapolation to 2100 |
| **Random Forest** | Captures non-linear patterns, lower variance | Higher accuracy on test set |

The `best_model.pkl` is automatically selected based on lowest RMSE on the held-out test set. Model metrics (MAE, RMSE, R²) are displayed on the [Analysis page](./templates/analysis.html).

---

## 📊 Dataset

**Source:** [Berkeley Earth Surface Temperature](http://berkeleyearth.org/data/)

| File | Coverage | Records |
|------|----------|---------|
| `GlobalTemperatures.csv` | 1750 – 2015 · monthly global land averages | ~3,200 rows |
| `CityTemperatures.csv` | City-level historical temperatures · 20+ cities | ~500k rows |

> Earth's average land surface temperature has risen approximately **+1.2 °C** since the pre-industrial baseline (1880–1900), with the rate of warming accelerating sharply after 1980.

---

## 🌐 Deploy on Hugging Face Spaces

This app is ready to deploy with the Docker SDK. The YAML metadata at the top of this README is already configured.

```python
# app.py — server binds to 0.0.0.0 on the Spaces-assigned port
port = int(os.environ.get("PORT", 7860))
app.run(host="0.0.0.0", port=port)
```

**Steps:**
1. Create a new Space → choose **Docker** as the SDK
2. Push this repo to the Space's git remote
3. Hugging Face builds the Docker image and serves on port 7860 automatically

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11, Flask 3.0 |
| **ML** | scikit-learn — `LinearRegression`, `RandomForestRegressor` |
| **Data** | pandas, NumPy |
| **Charts** | Chart.js 4 |
| **Weather API** | Open-Meteo (free, no API key required) |
| **Fonts** | Syne, Manrope, DM Mono (Google Fonts) |
| **Hosting** | Hugging Face Spaces (Docker) |

---

## 🗺 Supported Cities

`Hyderabad` · `Mumbai` · `Delhi` · `Bangalore` · `Chennai` · `Kolkata` · `Tokyo` · `Beijing` · `Shanghai` · `London` · `Paris` · `Berlin` · `New York` · `Los Angeles` · `Chicago` · `São Paulo` · `Lagos` · `Cairo` · `Sydney` · `Cape Town`

---

## 🤝 Contributing

Contributions are welcome! To get started:

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

Please keep PRs focused — one feature or fix per PR.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with 🌍 to make climate data accessible to everyone.

**[Live Demo](https://huggingface.co/spaces/sharmasai12/Unveiling-climate-change-dynamics-through-earth-surface-temperature-analysis)** · **[Report a Bug](https://huggingface.co/spaces/sharmasai12/Unveiling-climate-change-dynamics-through-earth-surface-temperature-analysis/discussions)** · **[Request a Feature](https://huggingface.co/spaces/sharmasai12/Unveiling-climate-change-dynamics-through-earth-surface-temperature-analysis/discussions)**

</div>
