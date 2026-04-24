"""
Climate Model — City-Level Temperature Training
================================================
Trains per-city regression models using city baseline offsets,
so predictions are realistic for each location (e.g. Hyderabad ~35°C in summer).
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))

# ─── City Registry ─────────────────────────────────────────────────────────
# lat, lon, and known monthly avg temps (°C) — ground-truth baselines
CITIES = {
    "Hyderabad":   {"lat": 17.38, "lon":  78.47, "country": "India",
                    "monthly_avg": [23,26,30,33,36,33,29,28,28,27,24,22]},
    "Mumbai":      {"lat": 19.08, "lon":  72.88, "country": "India",
                    "monthly_avg": [24,25,28,30,32,30,28,27,28,29,27,25]},
    "Delhi":       {"lat": 28.61, "lon":  77.21, "country": "India",
                    "monthly_avg": [14,17,23,30,36,39,35,34,32,27,20,15]},
    "Bangalore":   {"lat": 12.97, "lon":  77.59, "country": "India",
                    "monthly_avg": [20,23,27,29,28,24,23,23,23,23,21,19]},
    "Chennai":     {"lat": 13.08, "lon":  80.27, "country": "India",
                    "monthly_avg": [25,27,30,33,36,37,35,35,33,29,26,24]},
    "Kolkata":     {"lat": 22.57, "lon":  88.36, "country": "India",
                    "monthly_avg": [19,23,29,34,36,34,31,31,31,30,25,20]},
    "London":      {"lat": 51.51, "lon":  -0.13, "country": "UK",
                    "monthly_avg": [5,  6,  8, 11, 15, 18, 20, 20, 17, 13,  8,  5]},
    "New York":    {"lat": 40.71, "lon": -74.01, "country": "USA",
                    "monthly_avg": [0,  2,  7, 13, 19, 24, 27, 27, 22, 15,  9,  2]},
    "Tokyo":       {"lat": 35.68, "lon": 139.69, "country": "Japan",
                    "monthly_avg": [6,  7, 10, 15, 20, 23, 27, 29, 25, 19, 13,  8]},
    "Sydney":      {"lat":-33.87, "lon": 151.21, "country": "Australia",
                    "monthly_avg": [26, 26, 24, 21, 18, 15, 15, 16, 18, 20, 23, 25]},
    "Dubai":       {"lat": 25.20, "lon":  55.27, "country": "UAE",
                    "monthly_avg": [19, 21, 24, 28, 33, 35, 37, 38, 35, 31, 26, 21]},
    "Paris":       {"lat": 48.86, "lon":   2.35, "country": "France",
                    "monthly_avg": [5,  6,  9, 12, 16, 20, 23, 22, 18, 13,  8,  5]},
    "Singapore":   {"lat":  1.35, "lon": 103.82, "country": "Singapore",
                    "monthly_avg": [27, 28, 28, 29, 29, 29, 28, 28, 28, 28, 27, 27]},
    "Cairo":       {"lat": 30.06, "lon":  31.25, "country": "Egypt",
                    "monthly_avg": [13, 15, 18, 22, 27, 30, 31, 31, 28, 24, 19, 14]},
    "Moscow":      {"lat": 55.75, "lon":  37.62, "country": "Russia",
                    "monthly_avg": [-8, -7,  0,  9, 16, 20, 22, 21, 15,  7, -1, -6]},
    "Sao Paulo":   {"lat":-23.55, "lon": -46.63, "country": "Brazil",
                    "monthly_avg": [23, 23, 22, 20, 18, 17, 17, 18, 18, 20, 21, 22]},
    "Nairobi":     {"lat": -1.29, "lon":  36.82, "country": "Kenya",
                    "monthly_avg": [19, 20, 19, 18, 17, 16, 16, 16, 18, 19, 18, 18]},
    "Beijing":     {"lat": 39.91, "lon": 116.39, "country": "China",
                    "monthly_avg": [-3,  0,  6, 15, 21, 26, 29, 28, 22, 14,  5, -1]},
    "Los Angeles": {"lat": 34.05, "lon":-118.24, "country": "USA",
                    "monthly_avg": [14, 15, 16, 18, 20, 23, 26, 27, 25, 21, 17, 14]},
    "Lagos":       {"lat":  6.52, "lon":   3.38, "country": "Nigeria",
                    "monthly_avg": [28, 29, 29, 28, 28, 27, 26, 26, 27, 27, 28, 28]},
}

FEATURES = ["Year","Month","MonthSin","MonthCos","Season","Latitude","Longitude","LatAbs","IsTropical"]


def generate_city_dataset():
    """Generate realistic city-level monthly temperatures (1950-2023)."""
    np.random.seed(42)
    rows = []
    for city, info in CITIES.items():
        for year in range(1950, 2024):
            warming = (year - 1950) * 0.018   # ~1.3°C warming over 73 years
            for month in range(1, 13):
                base  = info["monthly_avg"][month - 1]
                noise = np.random.normal(0, 0.7)
                rows.append({
                    "City": city, "Country": info["country"],
                    "Latitude": info["lat"], "Longitude": info["lon"],
                    "Year": year, "Month": month,
                    "Temperature": round(base + warming + noise, 2),
                })
    df = pd.DataFrame(rows)
    os.makedirs(DATASET_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATASET_DIR, "CityTemperatures.csv"), index=False)
    print(f"[INFO] Generated city dataset: {len(df)} rows")
    return df


def add_features(df):
    df = df.copy()
    df["MonthSin"]   = np.sin(2 * np.pi * df["Month"] / 12)
    df["MonthCos"]   = np.cos(2 * np.pi * df["Month"] / 12)
    df["Season"]     = df["Month"].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
    df["LatAbs"]     = df["Latitude"].abs()
    df["IsTropical"] = (df["LatAbs"] < 23.5).astype(int)
    return df


def train():
    print("\n=== City-Level Climate Model Training ===\n")
    csv = os.path.join(DATASET_DIR, "CityTemperatures.csv")
    df  = pd.read_csv(csv) if os.path.exists(csv) else generate_city_dataset()
    df  = add_features(df)

    X = df[FEATURES]
    y = df["Temperature"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = Pipeline([("s", StandardScaler()), ("m", LinearRegression())])
    lr.fit(X_train, y_train)
    lp = lr.predict(X_test)
    lr_m = {"rmse": float(np.sqrt(mean_squared_error(y_test,lp))),
             "mae":  float(mean_absolute_error(y_test,lp)),
             "r2":   float(r2_score(y_test,lp))}
    print(f"  Linear Regression → RMSE:{lr_m['rmse']:.3f}  R²:{lr_m['r2']:.4f}")

    rf = Pipeline([("s", StandardScaler()), ("m", RandomForestRegressor(
            n_estimators=200, max_depth=14, random_state=42, n_jobs=-1))])
    rf.fit(X_train, y_train)
    rp = rf.predict(X_test)
    rf_m = {"rmse": float(np.sqrt(mean_squared_error(y_test,rp))),
             "mae":  float(mean_absolute_error(y_test,rp)),
             "r2":   float(r2_score(y_test,rp))}
    print(f"  Random Forest     → RMSE:{rf_m['rmse']:.3f}  R²:{rf_m['r2']:.4f}")

    best  = rf if rf_m["r2"] >= lr_m["r2"] else lr
    bname = "random_forest" if best is rf else "linear_regression"

    os.makedirs(MODEL_DIR, exist_ok=True)
    for obj, name in [(best,"best_model"),(lr,"lr_model"),(rf,"rf_model")]:
        with open(os.path.join(MODEL_DIR,f"{name}.pkl"),"wb") as f: pickle.dump(obj,f)

    metrics = {"linear_regression": lr_m, "random_forest": rf_m, "best": bname}
    with open(os.path.join(MODEL_DIR,"metrics.pkl"),"wb") as f: pickle.dump(metrics,f)
    with open(os.path.join(MODEL_DIR,"cities.pkl"),"wb") as f: pickle.dump(CITIES,f)

    print(f"\n[✓] Best model: {bname}  R²={max(lr_m['r2'],rf_m['r2']):.4f}")
    return best, metrics


def predict_city_temp(city: str, year: int, month: int, model=None):
    if model is None:
        with open(os.path.join(MODEL_DIR,"best_model.pkl"),"rb") as f:
            model = pickle.load(f)
    info = CITIES.get(city, CITIES["Hyderabad"])
    X = pd.DataFrame([{
        "Year": year, "Month": month,
        "MonthSin":   np.sin(2*np.pi*month/12),
        "MonthCos":   np.cos(2*np.pi*month/12),
        "Season":     {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[month],
        "Latitude":   info["lat"],  "Longitude": info["lon"],
        "LatAbs":     abs(info["lat"]),
        "IsTropical": int(abs(info["lat"]) < 23.5),
    }])
    return round(float(model.predict(X)[0]), 1)


if __name__ == "__main__":
    train()
