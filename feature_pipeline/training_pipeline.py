import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from dotenv import load_dotenv

# Fix Windows path issue
os.makedirs("C:\\tmp", exist_ok=True)
os.makedirs("D:\\tmp", exist_ok=True)

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Deep Learning
import keras
from keras import layers

import hopsworks

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
CERT_FOLDER       = "D:\\AQI_Predictor\\tmp"
MODEL_DIR         = "D:\\AQI_Predictor\\aqi-predictor\\models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Features used for training
FEATURE_COLS = [
    "pm25", "pm10", "no2", "o3",
    "temperature", "humidity", "wind_speed",
    "hour", "day_of_week", "month", "aqi_change_rate"
]
TARGET_COL = "aqi"


# ──────────────────────────────────────────────────────────────────────────────
# 1. FETCH FEATURES FROM HOPSWORKS
# ──────────────────────────────────────────────────────────────────────────────
def fetch_features():
    """
    Connects to Hopsworks and fetches all features from the feature store.
    Returns a pandas DataFrame.
    """
    os.makedirs(CERT_FOLDER, exist_ok=True)

    print("[Hopsworks] Connecting to Feature Store...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        cert_folder=CERT_FOLDER
    )
    fs = project.get_feature_store()

    print("[Hopsworks] Fetching features from 'aqi_features'...")
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)

    # Read all data as a DataFrame
    df = aqi_fg.read()
    print(f"[Hopsworks] Fetched {len(df)} rows successfully!")
    return df, project


# ──────────────────────────────────────────────────────────────────────────────
# 2. PREPARE DATA
# ──────────────────────────────────────────────────────────────────────────────
def prepare_data(df):
    """
    Cleans and prepares the DataFrame for training.
    Returns X_train, X_test, y_train, y_test, scaler.
    """
    print("\n[Data Prep] Preparing training data...")
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])

    # Fill remaining missing values with column mean
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].mean())

    # Sort by timestamp if available
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    print(f"  Features shape: {X.shape}")
    print(f"  Target shape:   {y.shape}")
    print(f"  AQI range: {y.min():.1f} — {y.max():.1f}")

    # Train/test split (80/20, no shuffle to preserve time order)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Scale features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler


# ──────────────────────────────────────────────────────────────────────────────
# 3. EVALUATE MODEL
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(model_name, y_test, y_pred):
    """Computes and prints RMSE, MAE, and R² for a model."""
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n  📊 {model_name} Results:")
    print(f"     RMSE : {rmse:.4f}")
    print(f"     MAE  : {mae:.4f}")
    print(f"     R²   : {r2:.4f}")

    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}


# ──────────────────────────────────────────────────────────────────────────────
# 4. TRAIN RANDOM FOREST
# ──────────────────────────────────────────────────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test):
    """Trains a Random Forest Regressor and returns model + metrics."""
    print("\n[Model 1] Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    metrics = evaluate("Random Forest", y_test, y_pred)

    # Save model
    path = os.path.join(MODEL_DIR, "random_forest.pkl")
    joblib.dump(model, path)
    print(f"  Saved to {path}")

    return model, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 5. TRAIN RIDGE REGRESSION
# ──────────────────────────────────────────────────────────────────────────────
def train_ridge(X_train, X_test, y_train, y_test):
    """Trains a Ridge Regression model and returns model + metrics."""
    print("\n[Model 2] Training Ridge Regression...")

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    metrics = evaluate("Ridge Regression", y_test, y_pred)

    # Save model
    path = os.path.join(MODEL_DIR, "ridge.pkl")
    joblib.dump(model, path)
    print(f"  Saved to {path}")

    return model, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 6. TRAIN NEURAL NETWORK (KERAS)
# ──────────────────────────────────────────────────────────────────────────────
def train_neural_network(X_train, X_test, y_train, y_test):
    """Trains a simple Neural Network using Keras."""
    print("\n[Model 3] Training Neural Network (Keras)...")

    n_features = X_train.shape[1]

    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.1,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
        ]
    )

    print(f"  Trained for {len(history.history['loss'])} epochs")

    y_pred  = model.predict(X_test, verbose=0).flatten()
    metrics = evaluate("Neural Network", y_test, y_pred)

    # Save model
    path = os.path.join(MODEL_DIR, "neural_network.keras")
    model.save(path)
    print(f"  Saved to {path}")

    return model, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 7. SAVE BEST MODEL TO HOPSWORKS MODEL REGISTRY
# ──────────────────────────────────────────────────────────────────────────────
def save_best_model(project, best_name, best_metrics, scaler):
    """
    Saves the best model and scaler to Hopsworks Model Registry.
    """
    print(f"\n[Registry] Saving best model ({best_name}) to Hopsworks...")

    mr = project.get_model_registry()

    # Save scaler alongside model
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # Save metrics as JSON
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)

    # Register model in Hopsworks
    aqi_model = mr.python.create_model(
        name="aqi_predictor",
        metrics={
            "rmse": round(best_metrics["rmse"], 4),
            "mae" : round(best_metrics["mae"],  4),
            "r2"  : round(best_metrics["r2"],   4),
        },
        description=f"Best AQI prediction model: {best_name}",
        input_example={"features": FEATURE_COLS},
        feature_view=None,
    )

    # Upload model files
    aqi_model.save(MODEL_DIR)
    print(f"[Registry] ✅ Model saved to Hopsworks Model Registry!")
    print(f"  View at: https://eu-west.cloud.hopsworks.ai:443/p/32036/models")


# ──────────────────────────────────────────────────────────────────────────────
# 8. MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def run_training_pipeline():
    print(f"\n{'='*55}")
    print(f"  AQI Predictor — Training Pipeline")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}")

    # Step 1: Fetch data
    df, project = fetch_features()

    # Step 2: Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Step 3: Train all models
    print(f"\n{'─'*55}")
    print("  Training Models")
    print(f"{'─'*55}")

    rf_model,  rf_metrics  = train_random_forest(X_train, X_test, y_train, y_test)
    ridge_model, ridge_metrics = train_ridge(X_train, X_test, y_train, y_test)
    nn_model,  nn_metrics  = train_neural_network(X_train, X_test, y_train, y_test)

    # Step 4: Compare models
    print(f"\n{'─'*55}")
    print("  Model Comparison")
    print(f"{'─'*55}")

    all_metrics = [rf_metrics, ridge_metrics, nn_metrics]

    print(f"\n  {'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print(f"  {'-'*52}")
    for m in all_metrics:
        print(f"  {m['model']:<25} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['r2']:>8.4f}")

    # Pick best model by lowest RMSE
    best = min(all_metrics, key=lambda x: x["rmse"])
    print(f"\n  🏆 Best Model: {best['model']} (RMSE: {best['rmse']:.4f})")

    # Step 5: Save best model to Hopsworks
    try:
        save_best_model(project, best["model"], best, scaler)
    except Exception as e:
        print(f"\n  ⚠️  Could not save to Model Registry: {e}")
        print(f"  Models are saved locally in: {MODEL_DIR}")

    print(f"\n{'='*55}")
    print(f"  ✅ TRAINING PIPELINE COMPLETE!")
    print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_training_pipeline()