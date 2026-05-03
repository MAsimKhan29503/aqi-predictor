import os
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")

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

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
MODEL_DIR         = "D:\\AQI_Predictor\\aqi-predictor\\models"
PLOTS_DIR         = "D:\\AQI_Predictor\\aqi-predictor\\notebooks\\plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURE_COLS = [
    "pm25", "pm10", "no2", "o3",
    "temperature", "humidity", "wind_speed",
    "hour", "day_of_week", "month", "aqi_change_rate"
]

# Set plot style
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"]      = 12


# ──────────────────────────────────────────────────────────────────────────────
# 1. FETCH DATA FROM HOPSWORKS
# ──────────────────────────────────────────────────────────────────────────────
def fetch_data():
    print("Connecting to Hopsworks...")
    os.makedirs(CERT_FOLDER, exist_ok=True)

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
        cert_folder=CERT_FOLDER
    )
    fs     = project.get_feature_store()
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
    df     = aqi_fg.read()

    print(f"Fetched {len(df)} rows from Hopsworks")
    return df, project


# ──────────────────────────────────────────────────────────────────────────────
# 2. BASIC EDA
# ──────────────────────────────────────────────────────────────────────────────
def run_eda(df):
    print("\n" + "="*55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*55)

    # Basic info
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"\n📋 Column Types:\n{df.dtypes}")
    print(f"\n📈 Basic Statistics:\n{df[FEATURE_COLS + ['aqi']].describe().round(2)}")
    print(f"\n🔍 Missing Values:\n{df.isnull().sum()}")

    # ── Plot 1: AQI Distribution ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AQI Distribution Analysis", fontsize=16, fontweight="bold")

    axes[0].hist(df["aqi"], bins=20, color="#ff6b6b", edgecolor="white", alpha=0.8)
    axes[0].set_title("AQI Frequency Distribution")
    axes[0].set_xlabel("AQI Value")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(df["aqi"].mean(), color="yellow", linestyle="--",
                    label=f"Mean: {df['aqi'].mean():.1f}")
    axes[0].axvline(df["aqi"].median(), color="cyan", linestyle="--",
                    label=f"Median: {df['aqi'].median():.1f}")
    axes[0].legend()

    axes[1].boxplot(df["aqi"], patch_artist=True,
                    boxprops=dict(facecolor="#ff6b6b", alpha=0.7))
    axes[1].set_title("AQI Box Plot")
    axes[1].set_ylabel("AQI Value")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_aqi_distribution.png"), dpi=150)
    plt.show()
    print("✅ Saved: 01_aqi_distribution.png")

    # ── Plot 2: AQI Over Time ─────────────────────────────────────────────────
    if "timestamp" in df.columns:
        df_sorted = df.sort_values("timestamp")

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(range(len(df_sorted)), df_sorted["aqi"],
                color="#ff6b6b", linewidth=1.5, alpha=0.8)
        ax.fill_between(range(len(df_sorted)), df_sorted["aqi"],
                        alpha=0.2, color="#ff6b6b")

        # AQI level zones
        ax.axhline(50,  color="#00e400", linestyle="--", alpha=0.5, label="Good (50)")
        ax.axhline(100, color="#ffff00", linestyle="--", alpha=0.5, label="Moderate (100)")
        ax.axhline(150, color="#ff7e00", linestyle="--", alpha=0.5, label="Unhealthy (150)")
        ax.axhline(200, color="#ff0000", linestyle="--", alpha=0.5, label="Very Unhealthy (200)")

        ax.set_title("AQI Trend Over Time", fontsize=16, fontweight="bold")
        ax.set_xlabel("Days")
        ax.set_ylabel("AQI")
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "02_aqi_trend.png"), dpi=150)
        plt.show()
        print("✅ Saved: 02_aqi_trend.png")

    # ── Plot 3: AQI by Month ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AQI Seasonal Patterns", fontsize=16, fontweight="bold")

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    monthly_aqi = df.groupby("month")["aqi"].mean().reset_index()
    monthly_aqi["month_name"] = monthly_aqi["month"].map(month_names)

    axes[0].bar(monthly_aqi["month_name"], monthly_aqi["aqi"],
                color="#4ecdc4", edgecolor="white", alpha=0.8)
    axes[0].set_title("Average AQI by Month")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Average AQI")

    hourly_aqi = df.groupby("hour")["aqi"].mean().reset_index()
    axes[1].plot(hourly_aqi["hour"], hourly_aqi["aqi"],
                 color="#45b7d1", linewidth=2, marker="o", markersize=6)
    axes[1].set_title("Average AQI by Hour of Day")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Average AQI")
    axes[1].set_xticks(range(0, 24, 2))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_aqi_seasonal.png"), dpi=150)
    plt.show()
    print("✅ Saved: 03_aqi_seasonal.png")

    # ── Plot 4: Correlation Heatmap ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))

    corr_cols = FEATURE_COLS + ["aqi"]
    corr      = df[corr_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.5, square=True)

    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_correlation_heatmap.png"), dpi=150)
    plt.show()
    print("✅ Saved: 04_correlation_heatmap.png")

    # ── Plot 5: AQI vs Weather Features ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("AQI vs Weather Features", fontsize=16, fontweight="bold")

    weather_features = ["temperature", "humidity", "wind_speed"]
    colors           = ["#ff6b6b", "#4ecdc4", "#45b7d1"]

    for i, (feat, color) in enumerate(zip(weather_features, colors)):
        axes[i].scatter(df[feat], df["aqi"], alpha=0.6, color=color, s=40)
        # Add trend line
        z = np.polyfit(df[feat].fillna(0), df["aqi"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
        axes[i].plot(x_line, p(x_line), "w--", linewidth=2, alpha=0.8)
        axes[i].set_xlabel(feat.replace("_", " ").title())
        axes[i].set_ylabel("AQI")
        axes[i].set_title(f"AQI vs {feat.replace('_', ' ').title()}")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_aqi_vs_weather.png"), dpi=150)
    plt.show()
    print("✅ Saved: 05_aqi_vs_weather.png")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. SHAP ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def run_shap_analysis(df):
    print("\n" + "="*55)
    print("  SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*55)

    # Load model and scaler
    rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    # Prepare data
    df_clean = df[FEATURE_COLS + ["aqi"]].dropna()
    X        = df_clean[FEATURE_COLS].values
    y        = df_clean["aqi"].values
    X_scaled = scaler.transform(X)

    feature_names = [f.replace("_", " ").title() for f in FEATURE_COLS]

    print(f"\nRunning SHAP analysis on {len(X)} samples...")

    # Create SHAP explainer
    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_scaled)

    # ── Plot 6: SHAP Summary Plot ─────────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_scaled,
        feature_names=feature_names,
        show=False,
        plot_size=(10, 7)
    )
    plt.title("SHAP Feature Importance Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_shap_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: 06_shap_summary.png")

    # ── Plot 7: SHAP Bar Plot ─────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_scaled,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance (Bar)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "07_shap_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: 07_shap_bar.png")

    # ── Print top features ────────────────────────────────────────────────────
    mean_shap   = np.abs(shap_values).mean(axis=0)
    shap_df     = pd.DataFrame({
        "Feature"   : feature_names,
        "SHAP Value": mean_shap
    }).sort_values("SHAP Value", ascending=False)

    print("\n📊 Top Features by SHAP Importance:")
    print(shap_df.to_string(index=False))

    return shap_df


# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL PERFORMANCE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def run_model_analysis(df):
    print("\n" + "="*55)
    print("  MODEL PERFORMANCE ANALYSIS")
    print("="*55)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    df_clean              = df[FEATURE_COLS + ["aqi"]].dropna()
    X                     = df_clean[FEATURE_COLS].values
    y                     = df_clean["aqi"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    X_test_scaled = scaler.transform(X_test)
    y_pred        = rf_model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

    # ── Plot 8: Actual vs Predicted ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Random Forest Model Performance", fontsize=16, fontweight="bold")

    axes[0].scatter(y_test, y_pred, alpha=0.7, color="#4ecdc4", s=60)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val],
                 "r--", linewidth=2, label="Perfect Prediction")
    axes[0].set_xlabel("Actual AQI")
    axes[0].set_ylabel("Predicted AQI")
    axes[0].set_title(f"Actual vs Predicted (R²={r2:.3f})")
    axes[0].legend()

    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=15, color="#ff6b6b",
                 edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="white", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Residual (Actual - Predicted)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "08_model_performance.png"), dpi=150)
    plt.show()
    print("✅ Saved: 08_model_performance.png")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AQI Predictor — EDA + SHAP Analysis")
    print("="*55)

    # Fetch data
    df, project = fetch_data()

    # Run EDA
    df = run_eda(df)

    # Run SHAP
    shap_df = run_shap_analysis(df)

    # Run model analysis
    run_model_analysis(df)

    print("\n" + "="*55)
    print("  ✅ EDA + SHAP ANALYSIS COMPLETE!")
    print(f"  All plots saved to: {PLOTS_DIR}")
    print("="*55)