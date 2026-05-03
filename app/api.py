import os
import platform
import joblib
import numpy as np
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import hopsworks

load_dotenv()

# ── Windows/Linux path fix ────────────────────────────────────────────────────
if platform.system() == "Windows":
    os.makedirs("C:\\tmp", exist_ok=True)
    os.makedirs("D:\\tmp", exist_ok=True)
    CERT_FOLDER = "D:\\AQI_Predictor\\tmp"
else:
    CERT_FOLDER = "/tmp"

# ── Config ────────────────────────────────────────────────────────────────────
HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT   = os.getenv("HOPSWORKS_PROJECT")
AQICN_API_KEY       = os.getenv("AQICN_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
MODEL_DIR           = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)

FEATURE_COLS = [
    "pm25", "pm10", "no2", "o3",
    "temperature", "humidity", "wind_speed",
    "hour", "day_of_week", "month", "aqi_change_rate"
]

AQI_LEVELS = [
    (0,   50,  "Good",                    "#00e400"),
    (51,  100, "Moderate",                "#ffff00"),
    (101, 150, "Unhealthy for Sensitive", "#ff7e00"),
    (151, 200, "Unhealthy",               "#ff0000"),
    (201, 300, "Very Unhealthy",          "#8f3f97"),
    (301, 500, "Hazardous",               "#7e0023"),
]

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────────────────────
model  = None
scaler = None

def load_model():
    """Loads model and scaler from local models directory."""
    global model, scaler
    try:
        model  = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        print(f"✅ Model loaded from {MODEL_DIR}")
    except Exception as e:
        print(f"⚠️  Could not load local model: {e}")
        print("Trying Hopsworks Model Registry...")
        try:
            os.makedirs(CERT_FOLDER, exist_ok=True)
            project = hopsworks.login(
                api_key_value=HOPSWORKS_API_KEY,
                project=HOPSWORKS_PROJECT,
                cert_folder=CERT_FOLDER
            )
            mr         = project.get_model_registry()
            hw_model   = mr.get_model("aqi_predictor", version=1)
            model_dir  = hw_model.download()
            model      = joblib.load(os.path.join(model_dir, "random_forest.pkl"))
            scaler     = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            print("✅ Model loaded from Hopsworks")
        except Exception as e2:
            print(f"❌ Could not load model: {e2}")


def get_aqi_level(aqi):
    """Returns label and color for a given AQI value."""
    for low, high, label, color in AQI_LEVELS:
        if low <= aqi <= high:
            return label, color
    return "Hazardous", "#7e0023"


def fetch_live_data():
    """Fetches current AQI and weather data."""
    # Fetch AQI
    aqi_url  = f"https://api.waqi.info/feed/islamabad/?token={AQICN_API_KEY}"
    aqi_data = {}
    try:
        r        = requests.get(aqi_url, timeout=10)
        data     = r.json()
        if data["status"] == "ok":
            iaqi     = data["data"]["iaqi"]
            aqi_data = {
                "aqi"       : float(data["data"]["aqi"]),
                "pm25"      : float(iaqi.get("pm25", {}).get("v", 0) or 0),
                "pm10"      : float(iaqi.get("pm10", {}).get("v", 0) or 0),
                "no2"       : float(iaqi.get("no2",  {}).get("v", 0) or 0),
                "o3"        : float(iaqi.get("o3",   {}).get("v", 0) or 0),
                "updated_at": data["data"]["time"]["s"],
                "station"   : data["data"]["city"]["name"],
                "forecast"  : data["data"].get("forecast", {}).get("daily", {}),
            }
    except Exception as e:
        print(f"AQI fetch error: {e}")

    # Fetch Weather
    weather_url  = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q=Islamabad,PK&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    weather_data = {}
    try:
        r            = requests.get(weather_url, timeout=10)
        data         = r.json()
        weather_data = {
            "temperature": float(data["main"]["temp"]),
            "humidity"   : float(data["main"]["humidity"]),
            "wind_speed" : float(data["wind"]["speed"]),
            "description": data["weather"][0]["description"].title(),
        }
    except Exception as e:
        print(f"Weather fetch error: {e}")

    return aqi_data, weather_data


# ──────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    """Root endpoint — API info."""
    return jsonify({
        "name"     : "AQI Predictor API",
        "version"  : "1.0",
        "city"     : "Islamabad / Rawalpindi",
        "endpoints": [
            "GET /health          — API health check",
            "GET /current        — Current AQI and weather",
            "GET /forecast       — 3-day AQI forecast",
            "GET /predict        — ML model prediction",
            "POST /predict/custom — Custom feature prediction",
        ]
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status"      : "healthy",
        "model_loaded": model is not None,
        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/current", methods=["GET"])
def current():
    """Returns current AQI and weather data."""
    aqi_data, weather_data = fetch_live_data()

    if not aqi_data:
        return jsonify({"error": "Could not fetch AQI data"}), 500

    current_aqi    = aqi_data["aqi"]
    label, color   = get_aqi_level(current_aqi)

    return jsonify({
        "city"        : "Islamabad",
        "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "aqi"         : current_aqi,
        "level"       : label,
        "color"       : color,
        "pollutants"  : {
            "pm25": aqi_data.get("pm25", 0),
            "pm10": aqi_data.get("pm10", 0),
            "no2" : aqi_data.get("no2",  0),
            "o3"  : aqi_data.get("o3",   0),
        },
        "weather"     : weather_data,
        "station"     : aqi_data.get("station", "Unknown"),
        "updated_at"  : aqi_data.get("updated_at", "Unknown"),
        "health_advice": (
            "Hazardous — Stay indoors!" if current_aqi > 300 else
            "Very Unhealthy — Avoid all outdoor activity" if current_aqi > 200 else
            "Unhealthy — Wear N95 mask outdoors" if current_aqi > 150 else
            "Unhealthy for Sensitive Groups — Limit exposure" if current_aqi > 100 else
            "Moderate — Unusually sensitive people should consider limiting prolonged exertion" if current_aqi > 50 else
            "Good — Air quality is satisfactory!"
        )
    })


@app.route("/forecast", methods=["GET"])
def forecast():
    """Returns 3-day AQI forecast."""
    aqi_data, weather_data = fetch_live_data()

    if not aqi_data:
        return jsonify({"error": "Could not fetch data"}), 500

    predictions  = []
    now          = datetime.now()
    pm25_forecast = aqi_data.get("forecast", {}).get("pm25", [])
    forecast_map  = {entry["day"]: entry["avg"] for entry in pm25_forecast}

    for i in range(1, 4):
        future_date   = now + timedelta(days=i)
        date_str      = future_date.strftime("%Y-%m-%d")

        if date_str in forecast_map:
            predicted_aqi = float(forecast_map[date_str])
            source        = "AQICN Forecast"
        elif model is not None and scaler is not None:
            features = np.array([[
                float(aqi_data.get("pm25", 0)),
                float(aqi_data.get("pm10", 0)),
                float(aqi_data.get("no2",  0)),
                float(aqi_data.get("o3",   0)),
                float(weather_data.get("temperature", 25)),
                float(weather_data.get("humidity", 50)),
                float(weather_data.get("wind_speed", 2)),
                float(future_date.hour),
                float(future_date.weekday()),
                float(future_date.month),
                0.0,
            ]])
            features_scaled = scaler.transform(features)
            predicted_aqi   = float(model.predict(features_scaled)[0])
            source          = "ML Model"
        else:
            predicted_aqi = float(aqi_data["aqi"])
            source        = "Current AQI (model unavailable)"

        label, color = get_aqi_level(predicted_aqi)
        predictions.append({
            "date"   : date_str,
            "day"    : future_date.strftime("%A"),
            "aqi"    : round(predicted_aqi),
            "level"  : label,
            "color"  : color,
            "source" : source,
        })

    return jsonify({
        "city"       : "Islamabad",
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "forecast"   : predictions,
    })


@app.route("/predict", methods=["GET"])
def predict():
    """Runs ML model prediction using current live data."""
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500

    aqi_data, weather_data = fetch_live_data()

    now      = datetime.now()
    features = np.array([[
        float(aqi_data.get("pm25", 0)),
        float(aqi_data.get("pm10", 0)),
        float(aqi_data.get("no2",  0)),
        float(aqi_data.get("o3",   0)),
        float(weather_data.get("temperature", 25)),
        float(weather_data.get("humidity", 50)),
        float(weather_data.get("wind_speed", 2)),
        float(now.hour),
        float(now.weekday()),
        float(now.month),
        0.0,
    ]])

    features_scaled = scaler.transform(features)
    predicted_aqi   = float(model.predict(features_scaled)[0])
    label, color    = get_aqi_level(predicted_aqi)

    return jsonify({
        "predicted_aqi": round(predicted_aqi),
        "level"        : label,
        "color"        : color,
        "timestamp"    : now.strftime("%Y-%m-%d %H:%M:%S"),
        "model"        : "Random Forest",
        "features_used": dict(zip(FEATURE_COLS, features[0].tolist())),
    })


@app.route("/predict/custom", methods=["POST"])
def predict_custom():
    """
    Custom prediction endpoint.
    Accepts JSON with feature values and returns AQI prediction.

    Example POST body:
    {
        "pm25": 80, "pm10": 40, "no2": 10, "o3": 5,
        "temperature": 30, "humidity": 45, "wind_speed": 3,
        "hour": 14, "day_of_week": 2, "month": 5,
        "aqi_change_rate": 0
    }
    """
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        features = np.array([[
            float(data.get("pm25",           0)),
            float(data.get("pm10",           0)),
            float(data.get("no2",            0)),
            float(data.get("o3",             0)),
            float(data.get("temperature",   25)),
            float(data.get("humidity",      50)),
            float(data.get("wind_speed",     2)),
            float(data.get("hour",           12)),
            float(data.get("day_of_week",     0)),
            float(data.get("month",           5)),
            float(data.get("aqi_change_rate", 0)),
        ]])

        features_scaled = scaler.transform(features)
        predicted_aqi   = float(model.predict(features_scaled)[0])
        label, color    = get_aqi_level(predicted_aqi)

        return jsonify({
            "predicted_aqi": round(predicted_aqi),
            "level"        : label,
            "color"        : color,
            "input_features": data,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  AQI Predictor — Flask API")
    print("="*50)
    load_model()
    print("\n  Endpoints:")
    print("  GET  http://localhost:5000/")
    print("  GET  http://localhost:5000/health")
    print("  GET  http://localhost:5000/current")
    print("  GET  http://localhost:5000/forecast")
    print("  GET  http://localhost:5000/predict")
    print("  POST http://localhost:5000/predict/custom")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)