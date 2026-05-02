import os
import platform

# Fix path issue - works on both Windows and Linux
if platform.system() == "Windows":
    os.makedirs("C:\\tmp", exist_ok=True)
    os.makedirs("D:\\tmp", exist_ok=True)
    os.environ["TMPDIR"] = "C:\\tmp"
    os.environ["TEMP"]   = "C:\\tmp"
    os.environ["TMP"]    = "C:\\tmp"
import requests
import pandas as pd
from datetime import datetime
import hopsworks
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
AQICN_API_KEY       = os.getenv("AQICN_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT   = os.getenv("HOPSWORKS_PROJECT")

# ── City Settings ─────────────────────────────────────────────────────────────
CITY            = "Islamabad"
CITY_AQICN      = "islamabad"          # used in AQICN URL
CITY_OPENWEATHER = "Islamabad,PK"      # used in OpenWeather API


# ──────────────────────────────────────────────────────────────────────────────
# 1. FETCH AQI DATA FROM AQICN
# ──────────────────────────────────────────────────────────────────────────────
def fetch_aqi_data():
    """
    Fetches current AQI and pollutant data from the AQICN API.
    Returns a dict with aqi, pm25, pm10, no2, o3 or None on failure.
    """
    url = f"https://api.waqi.info/feed/{CITY_AQICN}/?token={AQICN_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "ok":
            print(f"[AQICN] Bad status: {data['status']}")
            return None

        iaqi = data["data"]["iaqi"]   # individual AQI values

        return {
            "aqi"  : data["data"]["aqi"],
            "pm25" : iaqi.get("pm25", {}).get("v", None),
            "pm10" : iaqi.get("pm10", {}).get("v", None),
            "no2"  : iaqi.get("no2",  {}).get("v", None),
            "o3"   : iaqi.get("o3",   {}).get("v", None),
        }

    except Exception as e:
        print(f"[AQICN] Error fetching data: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 2. FETCH WEATHER DATA FROM OPENWEATHER
# ──────────────────────────────────────────────────────────────────────────────
def fetch_weather_data():
    """
    Fetches current weather data from OpenWeather API.
    Returns a dict with temperature, humidity, wind_speed or None on failure.
    """
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={CITY_OPENWEATHER}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            "temperature": data["main"]["temp"],
            "humidity"   : data["main"]["humidity"],
            "wind_speed" : data["wind"]["speed"],
        }

    except Exception as e:
        print(f"[OpenWeather] Error fetching data: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 3. COMPUTE FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def compute_features(aqi_data, weather_data, previous_aqi=None):
    """
    Combines raw AQI and weather data into a feature row.
    Adds time-based features and AQI change rate.
    Returns a pandas DataFrame with one row.
    """
    now = datetime.now()

    # Time-based features
    hour        = now.hour
    day_of_week = now.weekday()   # 0 = Monday, 6 = Sunday
    month       = now.month
    timestamp = pd.Timestamp(now)

    # AQI change rate (current - previous)
    aqi_change_rate = None
    if previous_aqi is not None:
        aqi_change_rate = aqi_data["aqi"] - previous_aqi

    # Build feature row
    feature_row = {
        "timestamp"      : timestamp,
    "city"           : CITY,
    "aqi"            : float(aqi_data["aqi"] or 0),
    "pm25"           : float(aqi_data["pm25"] or 0),
    "pm10"           : float(aqi_data["pm10"] or 0),
    "no2"            : float(aqi_data["no2"] or 0),
    "o3"             : float(aqi_data["o3"] or 0),
    "temperature"    : float(weather_data["temperature"] or 0),
    "humidity"       : float(weather_data["humidity"] or 0),
    "wind_speed"     : float(weather_data["wind_speed"] or 0),
    "hour"           : int(hour),
    "day_of_week"    : int(day_of_week),
    "month"          : int(month),
    "aqi_change_rate": float(aqi_change_rate or 0),
    }

    return pd.DataFrame([feature_row])


# ──────────────────────────────────────────────────────────────────────────────
# 4. STORE FEATURES IN HOPSWORKS FEATURE STORE
# ──────────────────────────────────────────────────────────────────────────────
def store_features(df):
    """
    Connects to Hopsworks and stores the feature DataFrame
    in a feature group called 'aqi_features'.
    """
    # Create cert folder in D drive
    cert_folder = "D:\\AQI_Predictor\\tmp" if platform.system() == "Windows" else "/tmp"
    os.makedirs(cert_folder, exist_ok=True)

    print("[Hopsworks] Connecting to Feature Store...")

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        cert_folder=cert_folder
    )
    fs = project.get_feature_store()

    # Get or create the feature group
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI and weather features for Islamabad",
        primary_key=["timestamp", "city"],
        event_time="timestamp",
    )

    print("[Hopsworks] Inserting features...")
    aqi_fg.insert(df)
    print("[Hopsworks] Features stored successfully!")


# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def run_feature_pipeline(previous_aqi=None):
    """
    Runs the full feature pipeline:
    1. Fetch AQI data
    2. Fetch weather data
    3. Compute features
    4. Store in Hopsworks
    """
    print(f"\n{'='*50}")
    print(f"Running Feature Pipeline at {datetime.now()}")
    print(f"{'='*50}")

    # Step 1: Fetch AQI data
    print("\n[Step 1] Fetching AQI data from AQICN...")
    aqi_data = fetch_aqi_data()
    if aqi_data is None:
        print("[ERROR] Could not fetch AQI data. Stopping pipeline.")
        return
    print(f"  AQI: {aqi_data['aqi']}, PM2.5: {aqi_data['pm25']}, PM10: {aqi_data['pm10']}")

    # Step 2: Fetch weather data
    print("\n[Step 2] Fetching weather data from OpenWeather...")
    weather_data = fetch_weather_data()
    if weather_data is None:
        print("[ERROR] Could not fetch weather data. Stopping pipeline.")
        return
    print(f"  Temp: {weather_data['temperature']}°C, Humidity: {weather_data['humidity']}%, Wind: {weather_data['wind_speed']} m/s")

    # Step 3: Compute features
    print("\n[Step 3] Computing features...")
    df = compute_features(aqi_data, weather_data, previous_aqi)
    print(df.to_string(index=False))

    # Step 4: Store in Hopsworks
    print("\n[Step 4] Storing features in Hopsworks...")
    store_features(df)

    print("\n[DONE] Feature pipeline completed successfully!")
    return aqi_data["aqi"]   # return current AQI for next run's change rate


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_feature_pipeline()