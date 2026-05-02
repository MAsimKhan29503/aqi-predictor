import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import hopsworks
from dotenv import load_dotenv
import time

# Fix Windows path issue - MUST be before hopsworks import
os.makedirs("C:\\tmp", exist_ok=True)
os.makedirs("D:\\tmp", exist_ok=True)

# Load environment variables
load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
AQICN_API_KEY     = os.getenv("AQICN_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

# ── Settings ──────────────────────────────────────────────────────────────────
CITY        = "Islamabad"
LAT         = 33.7215
LON         = 73.0433
CERT_FOLDER = "D:\\AQI_Predictor\\tmp"
DAYS_BACK   = 90


# ──────────────────────────────────────────────────────────────────────────────
# 1. FETCH HISTORICAL WEATHER FROM OPEN-METEO (FREE, NO API KEY)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_weather_bulk(start_date: datetime, end_date: datetime):
    """
    Fetches historical weather for a date range from Open-Meteo.
    Completely free, no API key required.
    Returns a dict of {date_str: {temperature, humidity, wind_speed}}
    """
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_str}&end_date={end_str}"
        f"&daily=temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max"
        f"&timezone=Asia%2FKarachi"
    )

    print(f"  Fetching weather data from Open-Meteo ({start_str} to {end_str})...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        daily    = data.get("daily", {})
        dates    = daily.get("time", [])
        temps    = daily.get("temperature_2m_mean", [])
        humidity = daily.get("relative_humidity_2m_mean", [])
        wind     = daily.get("wind_speed_10m_max", [])

        weather_by_date = {}
        for i, date in enumerate(dates):
            weather_by_date[date] = {
                "temperature": float(temps[i])    if temps[i]    is not None else 25.0,
                "humidity"   : float(humidity[i]) if humidity[i] is not None else 50.0,
                "wind_speed" : float(wind[i])     if wind[i]     is not None else 2.0,
            }

        print(f"  ✅ Got weather data for {len(weather_by_date)} days")
        return weather_by_date

    except Exception as e:
        print(f"  ❌ Open-Meteo error: {e}")
        print("  Using default weather values as fallback...")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# 2. FETCH HISTORICAL AQI FROM AQICN
# ──────────────────────────────────────────────────────────────────────────────
def fetch_historical_aqi():
    """
    Fetches available AQI forecast data from AQICN.
    Returns dict of {date_str: aqi_value}
    """
    url = f"https://api.waqi.info/feed/islamabad/?token={AQICN_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data     = response.json()

        if data["status"] != "ok":
            return {}

        forecast    = data["data"].get("forecast", {}).get("daily", {})
        pm25_data   = forecast.get("pm25", [])

        aqi_by_date = {}
        for entry in pm25_data:
            aqi_by_date[entry["day"]] = entry["avg"]

        return aqi_by_date
    except Exception as e:
        print(f"  [AQICN] Error: {e}")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# 3. GENERATE FEATURE ROW
# ──────────────────────────────────────────────────────────────────────────────
def generate_row(date: datetime, aqi_value: float,
                 weather: dict, previous_aqi: float = None):
    """
    Creates one feature row for a given date.
    Uses real AQI where available, otherwise realistic simulation.
    """
    import random

    if aqi_value is None:
        month    = date.month
        hour     = date.hour

        # Realistic Islamabad AQI by month
        base_aqi = {
            1: 160, 2: 150, 3: 120,
            4: 100, 5: 90,  6: 80,
            7: 70,  8: 75,  9: 85,
            10: 110, 11: 140, 12: 165
        }.get(month, 100)

        # Rush hour effect
        if hour in [7, 8, 9, 17, 18, 19]:
            base_aqi += random.randint(10, 30)
        elif hour in [2, 3, 4]:
            base_aqi -= random.randint(10, 20)

        aqi_value = max(30, base_aqi + random.randint(-20, 20))

    aqi_change_rate = float(aqi_value - previous_aqi) if previous_aqi else 0.0

    return {
        "timestamp"      : pd.Timestamp(date),
        "city"           : CITY,
        "aqi"            : float(aqi_value),
        "pm25"           : float(aqi_value),
        "pm10"           : float(aqi_value * 0.4),
        "no2"            : float(aqi_value * 0.1),
        "o3"             : float(aqi_value * 0.05),
        "temperature"    : float(weather.get("temperature", 25.0)),
        "humidity"       : float(weather.get("humidity", 50.0)),
        "wind_speed"     : float(weather.get("wind_speed", 2.0)),
        "hour"           : int(date.hour),
        "day_of_week"    : int(date.weekday()),
        "month"          : int(date.month),
        "aqi_change_rate": float(aqi_change_rate),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. CONNECT TO HOPSWORKS
# ──────────────────────────────────────────────────────────────────────────────
def connect_to_hopsworks():
    """Connects to Hopsworks and returns the feature group."""
    os.makedirs(CERT_FOLDER, exist_ok=True)

    print("\n[Hopsworks] Connecting...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        cert_folder=CERT_FOLDER
    )
    fs     = project.get_feature_store()
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="AQI and weather features for Islamabad",
        primary_key=["timestamp", "city"],
        event_time="timestamp",
    )
    print("[Hopsworks] Connected successfully!")
    return aqi_fg


# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN BACKFILL
# ──────────────────────────────────────────────────────────────────────────────
def run_backfill():
    print(f"\n{'='*55}")
    print(f"  AQI Historical Backfill — Last {DAYS_BACK} Days")
    print(f"{'='*55}\n")

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=DAYS_BACK)

    # Step 1: Fetch all weather data in one API call
    print("[Step 1] Fetching historical weather from Open-Meteo...")
    weather_by_date = fetch_weather_bulk(start_date, end_date)

    # Step 2: Fetch available real AQI data
    print("\n[Step 2] Fetching available AQI data from AQICN...")
    aqi_by_date = fetch_historical_aqi()
    print(f"  Found real AQI for {len(aqi_by_date)} dates: {list(aqi_by_date.keys())}")

    # Step 3: Generate all rows
    print(f"\n[Step 3] Generating {DAYS_BACK} days of features...")
    all_rows     = []
    previous_aqi = None
    current_date = start_date
    day_count    = 0

    while current_date <= end_date:
        day_count += 1
        date_str   = current_date.strftime("%Y-%m-%d")

        # Get real or simulated AQI
        real_aqi = aqi_by_date.get(date_str, None)

        # Get real or default weather
        weather = weather_by_date.get(date_str, {
            "temperature": 25.0,
            "humidity"   : 50.0,
            "wind_speed" : 2.0,
        })

        # Generate row
        row = generate_row(current_date, real_aqi, weather, previous_aqi)
        all_rows.append(row)
        previous_aqi = row["aqi"]

        if day_count % 10 == 0:
            print(f"  Processed {day_count}/{DAYS_BACK} days... "
                  f"{date_str} | AQI: {row['aqi']:.0f} | "
                  f"Temp: {row['temperature']:.1f}°C")

        current_date += timedelta(days=1)

    print(f"\n  ✅ Generated {len(all_rows)} feature rows")

    # Step 4: Create DataFrame
    print("\n[Step 4] Creating DataFrame...")
    df = pd.DataFrame(all_rows)
    print(f"  Shape: {df.shape}")
    print(f"\n  Sample (first 3 rows):")
    print(df[["timestamp", "aqi", "temperature", "humidity",
              "wind_speed", "month"]].head(3).to_string(index=False))

    # Step 5: Upload to Hopsworks
    print("\n[Step 5] Uploading to Hopsworks Feature Store...")
    aqi_fg = connect_to_hopsworks()
    aqi_fg.insert(df)

    print(f"\n{'='*55}")
    print(f"  ✅ BACKFILL COMPLETE!")
    print(f"  {len(df)} rows inserted into 'aqi_features'")
    print(f"  Date range: {start_date.date()} → {end_date.date()}")
    print(f"{'='*55}")
    print(f"\n  View data at:")
    print(f"  https://eu-west.cloud.hopsworks.ai:443/p/32036/fs/20720/fg/38868")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_backfill()