import os
import platform
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# Fix Windows/Linux path
if platform.system() == "Windows":
    os.makedirs("C:\\tmp", exist_ok=True)
    os.makedirs("D:\\tmp", exist_ok=True)
    CERT_FOLDER = "D:\\AQI_Predictor\\tmp"
else:
    CERT_FOLDER = "/tmp"

import hopsworks
import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
MODEL_DIR         = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)
PLOTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks", "plots")
)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURE_COLS = [
    "pm25", "pm10", "no2", "o3",
    "temperature", "humidity", "wind_speed",
    "hour", "day_of_week", "month", "aqi_change_rate"
]
TARGET_COL  = "aqi"
SEQ_LENGTH  = 7   # use 7 days of history to predict next day


# ──────────────────────────────────────────────────────────────────────────────
# 1. FETCH DATA FROM HOPSWORKS
# ──────────────────────────────────────────────────────────────────────────────
def fetch_data():
    print("[Hopsworks] Connecting...")
    os.makedirs(CERT_FOLDER, exist_ok=True)

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        cert_folder=CERT_FOLDER
    )
    fs     = project.get_feature_store()
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
    df     = aqi_fg.read()

    print(f"[Hopsworks] Fetched {len(df)} rows")
    return df, project


# ──────────────────────────────────────────────────────────────────────────────
# 2. PREPARE SEQUENCES FOR LSTM
# ──────────────────────────────────────────────────────────────────────────────
def prepare_sequences(df):
    """
    Prepares sequential data for LSTM.
    Creates sequences of SEQ_LENGTH days to predict the next day's AQI.
    """
    print(f"\n[Data Prep] Creating sequences of length {SEQ_LENGTH}...")

    # Sort by timestamp and clean
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    df = df[FEATURE_COLS + [TARGET_COL]].dropna()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].mean())

    # Scale all features + target together
    all_cols   = FEATURE_COLS + [TARGET_COL]
    scaler     = MinMaxScaler()
    scaled     = scaler.fit_transform(df[all_cols])

    # Save scaler for later use
    scaler_path = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")

    # Create sequences
    X, y = [], []
    for i in range(SEQ_LENGTH, len(scaled)):
        X.append(scaled[i - SEQ_LENGTH:i, :-1])  # features
        y.append(scaled[i, -1])                    # target (aqi)

    X = np.array(X)
    y = np.array(y)

    print(f"  Sequences shape: X={X.shape}, y={y.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, scaler, df


# ──────────────────────────────────────────────────────────────────────────────
# 3. BUILD LSTM MODEL
# ──────────────────────────────────────────────────────────────────────────────
def build_lstm(seq_length, n_features):
    """
    Builds a stacked LSTM model for AQI forecasting.
    """
    model = keras.Sequential([
        layers.Input(shape=(seq_length, n_features)),

        # First LSTM layer — return sequences for stacking
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),

        # Second LSTM layer
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),

        # Dense layers
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model


# ──────────────────────────────────────────────────────────────────────────────
# 4. TRAIN LSTM
# ──────────────────────────────────────────────────────────────────────────────
def train_lstm(X_train, X_test, y_train, y_test):
    """Trains the LSTM model with early stopping."""
    print("\n[LSTM] Building model...")

    seq_length = X_train.shape[1]
    n_features = X_train.shape[2]
    model      = build_lstm(seq_length, n_features)

    print(model.summary())

    print("\n[LSTM] Training...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=8,
        validation_split=0.1,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=7,
                verbose=1
            )
        ]
    )

    print(f"\n[LSTM] Trained for {len(history.history['loss'])} epochs")
    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# 5. EVALUATE LSTM
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_lstm(model, X_test, y_test, scaler, df):
    """Evaluates LSTM and converts predictions back to original AQI scale."""
    print("\n[LSTM] Evaluating...")

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform — need to reconstruct full array for inverse scaling
    n_features = len(FEATURE_COLS)

    def inverse_transform_target(scaled_values):
        dummy = np.zeros((len(scaled_values), n_features + 1))
        dummy[:, -1] = scaled_values
        return scaler.inverse_transform(dummy)[:, -1]

    y_test_actual = inverse_transform_target(y_test)
    y_pred_actual = inverse_transform_target(y_pred_scaled)

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae  = mean_absolute_error(y_test_actual, y_pred_actual)
    r2   = r2_score(y_test_actual, y_pred_actual)

    print(f"\n  📊 LSTM Results:")
    print(f"     RMSE : {rmse:.4f}")
    print(f"     MAE  : {mae:.4f}")
    print(f"     R²   : {r2:.4f}")

    return rmse, mae, r2, y_test_actual, y_pred_actual


# ──────────────────────────────────────────────────────────────────────────────
# 6. PLOT RESULTS
# ──────────────────────────────────────────────────────────────────────────────
def plot_results(history, y_test_actual, y_pred_actual, rmse, mae, r2):
    """Generates training history and prediction plots."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("LSTM Model — Training & Evaluation", fontsize=16, fontweight="bold")

    # Plot 1: Training Loss
    axes[0, 0].plot(history.history["loss"],     label="Train Loss", color="#4ecdc4")
    axes[0, 0].plot(history.history["val_loss"], label="Val Loss",   color="#ff6b6b")
    axes[0, 0].set_title("Training Loss Over Epochs")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].legend()

    # Plot 2: Training MAE
    axes[0, 1].plot(history.history["mae"],     label="Train MAE", color="#4ecdc4")
    axes[0, 1].plot(history.history["val_mae"], label="Val MAE",   color="#ff6b6b")
    axes[0, 1].set_title("Training MAE Over Epochs")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].legend()

    # Plot 3: Actual vs Predicted
    axes[1, 0].plot(y_test_actual, label="Actual AQI",    color="#4ecdc4", linewidth=2)
    axes[1, 0].plot(y_pred_actual, label="Predicted AQI", color="#ff6b6b",
                    linewidth=2, linestyle="--")
    axes[1, 0].set_title(f"Actual vs Predicted AQI (R²={r2:.3f})")
    axes[1, 0].set_xlabel("Test Sample")
    axes[1, 0].set_ylabel("AQI")
    axes[1, 0].legend()

    # Plot 4: Scatter
    axes[1, 1].scatter(y_test_actual, y_pred_actual, alpha=0.7, color="#45b7d1", s=60)
    min_val = min(y_test_actual.min(), y_pred_actual.min())
    max_val = max(y_test_actual.max(), y_pred_actual.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    axes[1, 1].set_title(f"Scatter: Actual vs Predicted\nRMSE={rmse:.2f}, MAE={mae:.2f}")
    axes[1, 1].set_xlabel("Actual AQI")
    axes[1, 1].set_ylabel("Predicted AQI")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "09_lstm_results.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"✅ Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. COMPARE WITH RANDOM FOREST
# ──────────────────────────────────────────────────────────────────────────────
def compare_models(lstm_rmse, lstm_mae, lstm_r2):
    """Compares LSTM with previously trained Random Forest."""
    print("\n" + "="*55)
    print("  Model Comparison")
    print("="*55)

    # Load RF metrics
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    try:
        with open(metrics_path) as f:
            rf_metrics = json.load(f)

        print(f"\n  {'Model':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
        print(f"  {'-'*47}")
        print(f"  {'Random Forest':<20} {rf_metrics['rmse']:>8.4f} "
              f"{rf_metrics['mae']:>8.4f} {rf_metrics['r2']:>8.4f}")
        print(f"  {'LSTM':<20} {lstm_rmse:>8.4f} {lstm_mae:>8.4f} {lstm_r2:>8.4f}")

        if lstm_rmse < rf_metrics["rmse"]:
            print(f"\n  🏆 LSTM wins with RMSE: {lstm_rmse:.4f}")
        else:
            print(f"\n  🏆 Random Forest wins with RMSE: {rf_metrics['rmse']:.4f}")
            print(f"  (LSTM needs more data to outperform tree-based models)")

    except Exception:
        print(f"  LSTM — RMSE: {lstm_rmse:.4f} | MAE: {lstm_mae:.4f} | R²: {lstm_r2:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. SAVE LSTM TO HOPSWORKS
# ──────────────────────────────────────────────────────────────────────────────
def save_lstm_to_hopsworks(project, model, lstm_rmse, lstm_mae, lstm_r2):
    """Saves LSTM model to Hopsworks Model Registry."""
    print("\n[Registry] Saving LSTM to Hopsworks...")
    try:
        # Save model locally first
        lstm_path = os.path.join(MODEL_DIR, "lstm_model.keras")
        model.save(lstm_path)
        print(f"  LSTM saved locally: {lstm_path}")

        mr         = project.get_model_registry()
        lstm_model = mr.python.create_model(
            name="aqi_lstm",
            metrics={
                "rmse": round(lstm_rmse, 4),
                "mae" : round(lstm_mae,  4),
                "r2"  : round(lstm_r2,   4),
            },
            description="LSTM deep learning model for AQI forecasting",
        )
        lstm_model.save(MODEL_DIR)
        print("[Registry] ✅ LSTM saved to Hopsworks!")

    except Exception as e:
        print(f"[Registry] ⚠️ Could not save to registry: {e}")
        print(f"  LSTM model saved locally at: {MODEL_DIR}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AQI Predictor — LSTM Training Pipeline")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)

    # Fetch data
    df, project = fetch_data()

    # Prepare sequences
    X_train, X_test, y_train, y_test, scaler, df = prepare_sequences(df)

    # Train LSTM
    model, history = train_lstm(X_train, X_test, y_train, y_test)

    # Evaluate
    rmse, mae, r2, y_test_actual, y_pred_actual = evaluate_lstm(
        model, X_test, y_test, scaler, df
    )

    # Plot results
    plot_results(history, y_test_actual, y_pred_actual, rmse, mae, r2)

    # Compare with Random Forest
    compare_models(rmse, mae, r2)

    # Save to Hopsworks
    save_lstm_to_hopsworks(project, model, rmse, mae, r2)

    print("\n" + "="*55)
    print("  ✅ LSTM TRAINING COMPLETE!")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)