import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.database.db import SessionLocal, DailyClimate, YieldHistory, DistrictBaselines
from src.features.phenology_clipper import SUPPORTED_CROPS, get_clipped_climate_data


def generate_training_data(db: Session, crop: str):
    """
    Builds the feature matrix for a single crop.

    Features: [compound_events, mean_ndvi, tmax_95th, sm_5th]
    Target  : yield_kg_ha

    compound_events = number of days in the critical window where
    Tmax > district 95th percentile AND soil_moisture < district 5th percentile
    (both thresholds are crop-specific / season-aware from DistrictBaselines).
    """
    print(f"  🧠 [{crop}] Building feature matrix...")
    features, labels = [], []

    yields = db.query(YieldHistory).filter(YieldHistory.crop == crop).all()
    if not yields:
        print(f"  ⚠️  [{crop}] No yield records found — skipping.")
        return np.array([]), np.array([])

    for y in yields:
        baseline = db.query(DistrictBaselines).filter(
            DistrictBaselines.district == y.district,
            DistrictBaselines.crop     == crop,
        ).first()
        if not baseline:
            continue

        climate = get_clipped_climate_data(db, y.district, y.year, crop)
        if not climate:
            continue

        # Extract features
        valid_tmax = [c.tmax for c in climate if c.tmax is not None]
        valid_sm = [c.soil_moisture for c in climate if c.soil_moisture is not None]
        valid_ndvi = [c.ndvi for c in climate if c.ndvi is not None]
        valid_precip = [c.precipitation for c in climate if c.precipitation is not None]

        mean_tmax = float(np.mean(valid_tmax)) if valid_tmax else 30.0
        mean_sm = float(np.mean(valid_sm)) if valid_sm else 0.15
        mean_ndvi = float(np.mean(valid_ndvi)) if valid_ndvi else 0.5
        tot_precip = float(np.sum(valid_precip)) if valid_precip else 0.0

        tmax_anomaly = mean_tmax - baseline.tmax_95th
        sm_anomaly = mean_sm - baseline.sm_5th

        features.append([mean_tmax, mean_sm, mean_ndvi, tot_precip, tmax_anomaly, sm_anomaly])
        labels.append(y.yield_kg_ha)

    return np.array(features), np.array(labels)


def train_model(crop: str):
    """Train and save one XGBoost Regressor for the given crop."""
    db   = SessionLocal()
    X, y = generate_training_data(db, crop)
    db.close()

    if len(X) < 4:
        print(f"  ⚠️  [{crop}] Not enough data to train ({len(X)} records). Skipping.")
        return

    print(f"  🚀 [{crop}] Training XGBoost on {len(X)} records...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        objective    = "reg:squarederror",
        n_estimators = 100,
        learning_rate= 0.1,
        max_depth    = 4,
        random_state = 42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    print(f"  📊 [{crop}] RMSE: {rmse:.1f} kg/ha")

    os.makedirs("models", exist_ok=True)
    path = f"models/xgboost_{crop.lower()}_model.pkl"
    joblib.dump(model, path)
    print(f"  ✅ [{crop}] Model saved → {path}")


def train_all_models():
    """Train one model per supported crop."""
    print("🌾 Starting multi-crop XGBoost training pipeline...")
    for crop in SUPPORTED_CROPS:
        train_model(crop)
    print("✅ All crop models trained.")


if __name__ == "__main__":
    train_all_models()
