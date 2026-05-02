import os
import platform
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import streamlit as st
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

FEATURE_COLS = [
    "pm25", "pm10", "no2", "o3",
    "temperature", "humidity", "wind_speed",
    "hour", "day_of_week", "month", "aqi_change_rate"
]

# ── AQI Level Definitions ─────────────────────────────────────────────────────
AQI_LEVELS = [
    (0,   50,  "Good",                    "#00e400", '<i class="fa-solid fa-face-smile"></i>'),
    (51,  100, "Moderate",                "#ffff00", '<i class="fa-solid fa-face-meh"></i>'),
    (101, 150, "Unhealthy for Sensitive", "#ff7e00", '<i class="fa-solid fa-head-side-mask"></i>'),
    (151, 200, "Unhealthy",               "#ff0000", '<i class="fa-solid fa-face-sad-tear"></i>'),
    (201, 300, "Very Unhealthy",          "#8f3f97", '<i class="fa-solid fa-face-frown"></i>'),
    (301, 500, "Hazardous",               "#7e0023", '<i class="fa-solid fa-skull-crossbones"></i>'),
]

def get_aqi_level(aqi):
    for low, high, label, color, emoji in AQI_LEVELS:
        if low <= aqi <= high:
            return label, color, emoji
    return "Hazardous", "#7e0023", "☠️"


# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_current_aqi():
    """Fetches current AQI from AQICN."""
    url = f"https://api.waqi.info/feed/islamabad/?token={AQICN_API_KEY}"
    try:
        r    = requests.get(url, timeout=10)
        data = r.json()
        if data["status"] == "ok":
            iaqi = data["data"]["iaqi"]
            return {
                "aqi"        : data["data"]["aqi"],
                "pm25"       : iaqi.get("pm25", {}).get("v", 0) or 0,
                "pm10"       : iaqi.get("pm10", {}).get("v", 0) or 0,
                "no2"        : iaqi.get("no2",  {}).get("v", 0) or 0,
                "o3"         : iaqi.get("o3",   {}).get("v", 0) or 0,
                "forecast"   : data["data"].get("forecast", {}).get("daily", {}),
                "station"    : data["data"]["city"]["name"],
                "updated_at" : data["data"]["time"]["s"],
            }
    except Exception as e:
        st.error(f"AQICN API error: {e}")
    return None


@st.cache_data(ttl=3600)
def fetch_current_weather():
    """Fetches current weather from OpenWeather."""
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q=Islamabad,PK&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    try:
        r    = requests.get(url, timeout=10)
        data = r.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity"   : data["main"]["humidity"],
            "wind_speed" : data["wind"]["speed"],
            "description": data["weather"][0]["description"].title(),
        }
    except Exception as e:
        st.error(f"OpenWeather API error: {e}")
    return None


@st.cache_resource
def load_model_and_scaler():
    """Loads model and scaler from Hopsworks Model Registry."""
    try:
        os.makedirs(CERT_FOLDER, exist_ok=True)
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT,
            cert_folder=CERT_FOLDER
        )
        mr    = project.get_model_registry()
        model = mr.get_model("aqi_predictor", version=1)

        model_dir = model.download()
        rf_model  = joblib.load(os.path.join(model_dir, "random_forest.pkl"))
        scaler    = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        return rf_model, scaler
    except Exception as e:
        st.warning(f"Could not load model from Hopsworks: {e}")
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────────────────────────────────────
def predict_next_3_days(aqi_data, weather_data, model, scaler):
    """
    Generates AQI predictions for the next 3 days.
    Uses real AQICN forecast data where available,
    otherwise uses the ML model.
    """
    predictions = []
    now         = datetime.now()

    # Try using real AQICN forecast first
    pm25_forecast = aqi_data.get("forecast", {}).get("pm25", [])
    forecast_map  = {entry["day"]: entry["avg"] for entry in pm25_forecast}

    for i in range(1, 4):
        future_date = now + timedelta(days=i)
        date_str    = future_date.strftime("%Y-%m-%d")

        # Use real forecast if available
        if date_str in forecast_map:
            predicted_aqi = forecast_map[date_str]
            source        = "AQICN Forecast"
        else:
            # Use ML model
            features = np.array([[
                float(aqi_data["pm25"]),
                float(aqi_data["pm10"]),
                float(aqi_data["no2"]),
                float(aqi_data["o3"]),
                float(weather_data["temperature"]),
                float(weather_data["humidity"]),
                float(weather_data["wind_speed"]),
                float(future_date.hour),
                float(future_date.weekday()),
                float(future_date.month),
                0.0,  # aqi_change_rate
            ]])

            if model is not None and scaler is not None:
                features_scaled = scaler.transform(features)
                predicted_aqi   = float(model.predict(features_scaled)[0])
                source          = "ML Model"
            else:
                predicted_aqi = float(aqi_data["aqi"])
                source        = "Current AQI"

        label, color, emoji = get_aqi_level(predicted_aqi)
        predictions.append({
            "date"         : future_date.strftime("%A, %b %d"),
            "date_obj"     : future_date,
            "aqi"          : round(predicted_aqi),
            "label"        : label,
            "color"        : color,
            "emoji"        : emoji,
            "source"       : source,
        })

    return predictions


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── Page config ───────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="AQI Predictor — Islamabad",
        page_icon=None,
        layout="wide",
    )

    # ── Custom CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        .aqi-card {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.5rem 0;
        }
        .metric-card {
            background: #1e1e2e;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        .forecast-card {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.5rem;
        }
        .alert-box {
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            font-size: 1.1rem;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1><i class="fa-solid fa-smog"></i> AQI Predictor</h1>
        <h3>Islamabad / Rawalpindi Air Quality Dashboard</h3>
        <p>Real-time monitoring and 3-day forecast</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
        aqi_data     = fetch_current_aqi()
        weather_data = fetch_current_weather()
        model, scaler = load_model_and_scaler()

    if not aqi_data or not weather_data:
        st.error("Could not fetch live data. Please check your API keys.")
        return

    current_aqi         = aqi_data["aqi"]
    label, color, emoji = get_aqi_level(current_aqi)

    # ── ALERT BANNER ──────────────────────────────────────────────────────────
    if current_aqi > 150:
        st.markdown(f"""
        <div class="alert-box" style="background-color:#ff000033; border:2px solid #ff0000;">
            <i class="fa-solid fa-bell"></i> <b>HEALTH ALERT:</b> AQI is {current_aqi} — {label}!
            Avoid outdoor activities. Wear N95 mask if going outside.
        </div>
        """, unsafe_allow_html=True)
    elif current_aqi > 100:
        st.markdown(f"""
        <div class="alert-box" style="background-color:#ff7e0033; border:2px solid #ff7e00;">
            <i class="fa-solid fa-triangle-exclamation"></i> <b>CAUTION:</b> AQI is {current_aqi} — {label}.
            Sensitive groups should limit outdoor exposure.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-box" style="background-color:#00e40033; border:2px solid #00e400;">
            <i class="fa-solid fa-circle-check"></i> <b>AIR QUALITY:</b> AQI is {current_aqi} — {label}. Enjoy your day!
        </div>
        """, unsafe_allow_html=True)

    # ── Current AQI + Weather ─────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="aqi-card" style="background-color:{color}33; border:2px solid {color};">
            <h2>{emoji}</h2>
            <h1 style="color:{color}; font-size:3rem;">{current_aqi}</h1>
            <p><b>AQI</b></p>
            <p style="color:{color};">{label}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("Temperature", f"{weather_data['temperature']:.1f}°C")
        st.metric("Humidity", f"{weather_data['humidity']}%")

    with col3:
        st.metric("Wind Speed", f"{weather_data['wind_speed']} m/s")
        st.metric("Weather", weather_data['description'])

    with col4:
        st.metric("PM2.5", f"{aqi_data['pm25']:.0f} µg/m³")
        st.metric("PM10",  f"{aqi_data['pm10']:.0f} µg/m³")

    with col5:
        st.metric("NO₂", f"{aqi_data['no2']:.0f} µg/m³")
        st.metric("O₃",  f"{aqi_data['o3']:.0f} µg/m³")

    st.markdown(f"<div style='color:gray'>"
                f"<i class='fa-solid fa-location-dot'></i> Station: {aqi_data['station']} | "
                f"<i class='fa-solid fa-clock'></i> Updated: {aqi_data['updated_at']}"
                f"</div>", unsafe_allow_html=True)

    # ── Data Staleness Check ──────────────────────────────────────────────────
    try:
        updated_dt = datetime.strptime(aqi_data['updated_at'], "%Y-%m-%d %H:%M:%S")
        hours_old = (datetime.now() - updated_dt).total_seconds() / 3600
        days_old = hours_old / 24
        
        if hours_old > 168:  # 7 days
            st.error(f"Station appears offline. Data is {hours_old:.0f} hours ({days_old:.1f} days) old.")
        elif hours_old > 24:
            st.warning(f"Data is stale ({hours_old:.1f} hours old), but OpenWeather data is real-time.")
        else:
            st.success(f"Data is fresh ({hours_old:.1f} hours old).")
    except Exception as e:
        st.warning(f"Could not parse data timestamp: {e}")

    st.divider()

    # ── 3-Day Forecast ────────────────────────────────────────────────────────
    st.subheader("3-Day AQI Forecast")

    predictions = predict_next_3_days(aqi_data, weather_data, model, scaler)

    cols = st.columns(3)
    for i, pred in enumerate(predictions):
        with cols[i]:
            st.markdown(f"""
            <div class="forecast-card"
                 style="background-color:{pred['color']}22;
                        border: 2px solid {pred['color']};">
                <h3>{pred['date']}</h3>
                <h1 style="color:{pred['color']}; font-size:2.5rem;">
                    {pred['emoji']} {pred['aqi']}
                </h1>
                <p><b>{pred['label']}</b></p>
                <small>Source: {pred['source']}</small>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── AQI Trend Chart ───────────────────────────────────────────────────────
    st.subheader("AQI Trend (Current + 3-Day Forecast)")

    chart_data = pd.DataFrame([
        {"Date": datetime.now().strftime("%b %d (Now)"), "AQI": current_aqi},
        *[{"Date": p["date"], "AQI": p["aqi"]} for p in predictions]
    ])

    st.line_chart(chart_data.set_index("Date"))

    st.divider()

    # ── AQI Scale Reference ───────────────────────────────────────────────────
    st.subheader("AQI Scale Reference")

    scale_cols = st.columns(6)
    for i, (low, high, label, color, emoji) in enumerate(AQI_LEVELS):
        with scale_cols[i]:
            st.markdown(f"""
            <div style="background-color:{color}33;
                        border:2px solid {color};
                        padding:0.8rem;
                        border-radius:8px;
                        text-align:center;">
                <b>{emoji}</b><br>
                <b style="color:{color};">{low}–{high}</b><br>
                <small>{label}</small>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; color:gray; padding:1rem;">
        <p>🌍 Data sources: AQICN · OpenWeather · Open-Meteo</p>
        <p>🤖 ML Model: Random Forest trained on Islamabad AQI data</p>
        <p>⚡ Powered by Hopsworks Feature Store · GitHub Actions</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()