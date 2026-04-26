import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy.orm import Session
from src.database.db import SessionLocal, DailyClimate, YieldHistory, DistrictBaselines
from src.features.phenology_clipper import get_clipped_climate_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def generate_training_data(db: Session):
    """
    Simulates joining clipped climate data with Yield outcomes.
    Calculates the 'Compound Stress Feature':
    1 if (Tmax > 95th) AND (Soil Moisture < 5th) else 0.
    """
    print("🧠 Generating Feature Matrix for XGBoost...")
    features = []
    labels = []
    
    # We will use mock yields and mock climate data
    yields = db.query(YieldHistory).all()
    
    for y in yields:
        # Get baseline for this district
        baseline = db.query(DistrictBaselines).filter(DistrictBaselines.district == y.district).first()
        if not baseline:
            continue
            
        # Get phenology clipped climate data for that year (mocking year by just getting all for district)
        climate = get_clipped_climate_data(db, y.district, 2026, y.crop) # In reality, use y.year
        
        # Calculate Compound Stress Events
        # Number of days in the critical window where BOTH heat > 95th and SM < 5th
        compound_events = 0
        for c in climate:
            if c.tmax > baseline.tmax_95th and c.soil_moisture < baseline.sm_5th:
                compound_events += 1
                
        # Calculate mean NDVI as a feature
        mean_ndvi = np.mean([c.ndvi for c in climate]) if climate else 0.5
        
        features.append([
            compound_events,
            mean_ndvi,
            baseline.tmax_95th,
            baseline.sm_5th
        ])
        labels.append(y.yield_kg_ha)
        
    return np.array(features), np.array(labels)

def train_model():
    db = SessionLocal()
    X, y = generate_training_data(db)
    db.close()
    
    if len(X) == 0:
        print("⚠️ Not enough data to train. Returning mock model.")
        return
        
    print(f"🚀 Training XGBoost on {len(X)} historical records...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"📊 Model trained. MSE: {mse:.2f}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_csi_model.pkl")
    print("✅ Model saved to models/xgboost_csi_model.pkl")

if __name__ == "__main__":
    train_model()
