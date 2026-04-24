"""
Unveiling Climate Change Dynamics Through Earth Surface Temperature Analysis Flask Application
"""
import os, pickle, json, datetime
import numpy as np
import pandas as pd
import urllib.request
from flask import Flask, render_template, request, jsonify

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

app = Flask(__name__, template_folder="templates", static_folder="static")

def _load(name):
    p = os.path.join(MODEL_DIR, name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

city_best    = _load("best_model.pkl")
city_lr      = _load("lr_model.pkl")
city_rf      = _load("rf_model.pkl")
city_metrics = _load("metrics.pkl")
CITIES       = _load("cities.pkl") or {}
glob_best    = _load("global_best_model.pkl")
glob_lr      = _load("global_lr_model.pkl")
glob_rf      = _load("global_rf_model.pkl")
glob_metrics = _load("global_metrics.pkl")

MONTH_NAMES  = ["January","February","March","April","May","June",
                "July","August","September","October","November","December"]
SEASON_NAMES = ["Winter","Spring","Summer","Autumn"]
SEASON_MAP   = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}

WEATHER_CODES = {
    0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
    45:"Foggy",48:"Icy fog",51:"Light drizzle",53:"Drizzle",55:"Heavy drizzle",
    61:"Slight rain",63:"Moderate rain",65:"Heavy rain",71:"Slight snow",
    73:"Moderate snow",75:"Heavy snow",80:"Rain showers",81:"Moderate showers",
    82:"Heavy showers",95:"Thunderstorm",96:"Thunderstorm + hail",
}

def city_features(city, year, month):
    info = CITIES.get(city, {"lat": 17.38, "lon": 78.47})
    return pd.DataFrame([{
        "Year": year, "Month": month,
        "MonthSin":   np.sin(2*np.pi*month/12),
        "MonthCos":   np.cos(2*np.pi*month/12),
        "Season":     SEASON_MAP[month],
        "Latitude":   info["lat"], "Longitude": info["lon"],
        "LatAbs":     abs(info["lat"]),
        "IsTropical": int(abs(info["lat"]) < 23.5),
    }])

def global_features(year, month):
    base = 9.0 + (year - 1900) * 0.010
    return pd.DataFrame([{
        "Year": year, "Month": month,
        "MonthSin":  np.sin(2*np.pi*month/12),
        "MonthCos":  np.cos(2*np.pi*month/12),
        "Season":    SEASON_MAP[month],
        "Rolling12": base,
        "TempLag1":  base + 2.5*np.sin(2*np.pi*(month-2)/12),
        "TempLag12": base,
    }])

def fetch_live_weather(lat, lon):
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}&current_weather=true"
           f"&hourly=relative_humidity_2m,apparent_temperature&timezone=auto&forecast_days=1")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ClimateChangeDynamics/4.0"})
        with urllib.request.urlopen(req, timeout=7) as r:
            data = json.loads(r.read().decode())
        cw = data.get("current_weather", {})
        hourly = data.get("hourly", {})
        return {
            "temp":        round(cw.get("temperature", 0), 1),
            "feels_like":  round(hourly.get("apparent_temperature", [None])[0] or 0, 1),
            "humidity":    hourly.get("relative_humidity_2m", [None])[0],
            "windspeed":   round(cw.get("windspeed", 0), 1),
            "is_day":      cw.get("is_day", 1),
            "code":        cw.get("weathercode", 0),
            "description": WEATHER_CODES.get(cw.get("weathercode", 0), "Unknown"),
        }
    except Exception:
        return None

def get_chart_data():
    csv = os.path.join(DATASET_DIR, "GlobalTemperatures.csv")
    if not os.path.exists(csv):
        return {}
    df = pd.read_csv(csv)
    df["dt"]    = pd.to_datetime(df["dt"])
    df["Year"]  = df["dt"].dt.year
    df["Month"] = df["dt"].dt.month
    df.dropna(subset=["LandAverageTemperature"], inplace=True)
    annual  = df.groupby("Year")["LandAverageTemperature"].mean().reset_index()
    df["Decade"] = (df["Year"] // 10) * 10
    decade  = df.groupby("Decade")["LandAverageTemperature"].mean().reset_index()
    monthly = df.groupby("Month")["LandAverageTemperature"].mean().reset_index()
    return {
        "annual_years":  annual["Year"].tolist(),
        "annual_temps":  [round(v,3) for v in annual["LandAverageTemperature"].tolist()],
        "decade_labels": [str(int(d))+"s" for d in decade["Decade"].tolist()],
        "decade_temps":  [round(v,3) for v in decade["LandAverageTemperature"].tolist()],
        "month_labels":  ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        "month_temps":   [round(v,3) for v in monthly["LandAverageTemperature"].tolist()],
        "stats": {
            "min_year":   int(df["Year"].min()), "max_year": int(df["Year"].max()),
            "total_rows": len(df),
            "min_temp":   round(float(df["LandAverageTemperature"].min()), 2),
            "max_temp":   round(float(df["LandAverageTemperature"].max()), 2),
            "mean_temp":  round(float(df["LandAverageTemperature"].mean()), 2),
        }
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html",
        chart_data=get_chart_data(), metrics={"city": city_metrics, "global": glob_metrics})

@app.route("/predict", methods=["GET","POST"])
def predict():
    city_result=None; global_result=None; error=None
    tab = request.form.get("tab","city")
    if request.method == "POST":
        try:
            year  = int(request.form.get("year",  2025))
            month = int(request.form.get("month", 5))
            if not (1 <= month <= 12):
                raise ValueError("Invalid month.")
            if tab == "global":
                if not (1900 <= year <= 2100):
                    raise ValueError("Year must be between 1900 and 2100.")
                Xg = global_features(year, month)
                g_lr_p   = round(float(glob_lr.predict(Xg)[0]),   2) if glob_lr   else None
                g_rf_p   = round(float(glob_rf.predict(Xg)[0]),   2) if glob_rf   else None
                g_best_p = round(float(glob_best.predict(Xg)[0]), 2) if glob_best else None
                g_baseline = 8.6
                global_result = {
                    "year": year, "month": MONTH_NAMES[month-1],
                    "season": SEASON_NAMES[SEASON_MAP[month]],
                    "lr_pred": g_lr_p, "rf_pred": g_rf_p,
                    "best_pred": g_best_p, "baseline": g_baseline,
                    "anomaly": round(g_best_p - g_baseline, 2) if g_best_p else None,
                    "meaning": "Average of ALL Earth land points, from Arctic and Siberia to deserts and tropics. Range: 6C to 14C.",
                }
                return render_template("predict.html",
                    city_result=city_result, global_result=global_result,
                    error=error, tab=tab, cities=sorted(CITIES.keys()), CITIES=CITIES)
            city = request.form.get("city","Hyderabad")
            if city not in CITIES:
                raise ValueError(f"City '{city}' not supported.")
            if not (1950 <= year <= 2100):
                raise ValueError("Year must be between 1950 and 2100.")
            Xc = city_features(city, year, month)
            c_lr_p   = round(float(city_lr.predict(Xc)[0]),   1) if city_lr   else None
            c_rf_p   = round(float(city_rf.predict(Xc)[0]),   1) if city_rf   else None
            c_best_p = round(float(city_best.predict(Xc)[0]), 1) if city_best else None
            c_baseline = CITIES[city]["monthly_avg"][month-1]
            c_anomaly  = round(c_best_p - c_baseline, 1) if c_best_p else None
            info = CITIES[city]
            live = fetch_live_weather(info["lat"], info["lon"]) if year <= datetime.date.today().year else None
            city_result = {
                "city": city, "country": info["country"],
                "year": year, "month": MONTH_NAMES[month-1],
                "season": SEASON_NAMES[SEASON_MAP[month]],
                "lr_pred": c_lr_p, "rf_pred": c_rf_p,
                "best_pred": c_best_p, "baseline": c_baseline,
                "anomaly": c_anomaly, "live": live,
                "lat": info["lat"], "lon": info["lon"],
            }
            g_year = max(1900, min(2100, year))
            Xg = global_features(g_year, month)
            g_lr_p   = round(float(glob_lr.predict(Xg)[0]),   2) if glob_lr   else None
            g_rf_p   = round(float(glob_rf.predict(Xg)[0]),   2) if glob_rf   else None
            g_best_p = round(float(glob_best.predict(Xg)[0]), 2) if glob_best else None
            g_baseline = 8.6
            global_result = {
                "year": g_year, "month": MONTH_NAMES[month-1],
                "season": SEASON_NAMES[SEASON_MAP[month]],
                "lr_pred": g_lr_p, "rf_pred": g_rf_p,
                "best_pred": g_best_p, "baseline": g_baseline,
                "anomaly": round(g_best_p - g_baseline, 2) if g_best_p else None,
                "meaning": "Average of ALL Earth land — Arctic, Siberia, deserts & tropics. Range: 6°C–14°C.",
            }
        except Exception as exc:
            error = str(exc)
    return render_template("predict.html",
        city_result=city_result, global_result=global_result,
        error=error, tab=tab, cities=sorted(CITIES.keys()), CITIES=CITIES)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/api/predict", methods=["POST"])
def api_city_predict():
    d=request.get_json(force=True); city=d.get("city","Hyderabad")
    year=int(d.get("year",2030)); month=int(d.get("month",6))
    X=city_features(city,year,month)
    pred=round(float(city_best.predict(X)[0]),1) if city_best else None
    return jsonify({"city":city,"year":year,"month":month,"predicted_temp_c":pred})

@app.route("/api/global", methods=["POST"])
def api_global_predict():
    d=request.get_json(force=True); year=int(d.get("year",2030)); month=int(d.get("month",6))
    X=global_features(year,month)
    pred=round(float(glob_best.predict(X)[0]),2) if glob_best else None
    return jsonify({"year":year,"month":month,"predicted_global_avg_c":pred})

@app.route("/api/weather")
def api_weather():
    city=request.args.get("city","Hyderabad"); info=CITIES.get(city)
    if not info: return jsonify({"error":"City not found"}),404
    live=fetch_live_weather(info["lat"],info["lon"])
    if live: return jsonify({"city":city,"country":info["country"],"lat":info["lat"],"lon":info["lon"],"weather":live})
    return jsonify({"error":"Live weather unavailable"}),503

@app.route("/api/cities")
def api_cities():
    return jsonify({c:{"lat":v["lat"],"lon":v["lon"],"country":v["country"],"monthly_avg":v["monthly_avg"]} for c,v in CITIES.items()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
