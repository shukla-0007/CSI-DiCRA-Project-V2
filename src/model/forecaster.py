import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from src.database.db import SessionLocal, DistrictBaselines, RiskPredictions
from datetime import datetime

def load_cmip6_mock_data(scenario: str):
    """
    Mock function to represent loading downscaled CMIP6 data 
    (from IITM Pune / CCKP) for a specific RCP scenario.
    """
    print(f"🌍 Loading CMIP6 {scenario} Data (2026-2041)...")
    
    # In reality, this reads a NetCDF or large CSV
    years = list(range(2026, 2042))
    districts = [
        "Agar Malwa", "Alirajpur", "Anuppur", "Ashoknagar", "Balaghat", "Barwani", "Betul", "Bhind", 
        "Bhopal", "Burhanpur", "Chhatarpur", "Chhindwara", "Damoh", "Datia", "Dewas", "Dhar", 
        "Dindori", "Guna", "Gwalior", "Harda", "Hoshangabad", "Indore", "Jabalpur", "Jhabua", 
        "Katni", "Khandwa", "Khargone", "Mandla", "Mandsaur", "Morena", "Narsinghpur", "Neemuch", 
        "Niwari", "Panna", "Raisen", "Rajgarh", "Ratlam", "Rewa", "Sagar", "Satna", "Sehore", 
        "Seoni", "Shahdol", "Shajapur", "Sheopur", "Shivpuri", "Sidhi", "Singrauli", "Tikamgarh", 
        "Ujjain", "Umaria", "Vidisha"
    ]
    
    data = [] 
    for d in districts:
        for y in years:
            # Simulate compound events building up over time in RCP 8.5 vs 4.5
            base_events = 5
            multiplier = 1.5 if scenario == "SSP5_8.5" else 1.1
            future_events = int(base_events * multiplier * (y - 2025) / 5)
            
            data.append({
                "district": d,
                "year": y,
                "compound_events": max(0, min(15, future_events)),
                "mean_ndvi": np.random.uniform(0.3, 0.7) # Simulating future NDVI drops
            })
    return pd.DataFrame(data)

def generate_forecasts():
    db = SessionLocal()
    
    model_path = "models/xgboost_csi_model.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}. Train it first.")
        return
        
    model = joblib.load(model_path)
    print("🤖 Model loaded. Generating 15-year risk trajectories...")
    
    for scenario in ["SSP2_4.5", "SSP5_8.5"]:
        cmip6_df = load_cmip6_mock_data(scenario)
        
        for _, row in cmip6_df.iterrows():
            # Get the static historical baselines
            baseline = db.query(DistrictBaselines).filter(DistrictBaselines.district == row['district']).first()
            if not baseline:
                continue
                
            features = np.array([[
                row['compound_events'],
                row['mean_ndvi'],
                baseline.tmax_95th,
                baseline.sm_5th
            ]])
            
            # Predict yield drop
            predicted_yield = model.predict(features)[0]
            
            # Convert predicted yield drop to a 1-5 Risk Tier probability
            # (Mock conversion logic: lower yield = higher risk)
            if predicted_yield < 700:
                tier = 5
            elif predicted_yield < 850:
                tier = 4
            elif predicted_yield < 1000:
                tier = 3
            elif predicted_yield < 1150:
                tier = 2
            else:
                tier = 1
                
            # Upsert into RiskPredictions
            existing = db.query(RiskPredictions).filter(
                RiskPredictions.district == row['district'],
                RiskPredictions.year == row['year'],
                RiskPredictions.scenario == scenario
            ).first()
            
            if existing:
                existing.risk_tier = tier
                existing.csi_score = tier / 5.0
            else:
                record = RiskPredictions(
                    district=row['district'],
                    year=row['year'],
                    scenario=scenario,
                    csi_score=tier / 5.0,
                    risk_tier=tier
                )
                db.add(record)
                
    db.commit()
    print("✅ All future risk trajectories saved to database.")
    db.close()

if __name__ == "__main__":
    generate_forecasts()
