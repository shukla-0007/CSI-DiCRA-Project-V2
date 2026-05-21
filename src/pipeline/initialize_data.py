import os
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from sqlalchemy import insert
from src.database.db import init_db, SessionLocal, DailyClimate, YieldHistory
from src.pipeline.pmfby_validation import sync_yield_history
from src.features.dynamic_baselines import calculate_district_baselines
from src.features.phenology_clipper import SUPPORTED_CROPS
from src.model.train_xgboost import train_all_models
from src.model.forecaster import generate_forecasts

MP_DISTRICTS = [
    "Agar Malwa", "Alirajpur", "Anuppur", "Ashoknagar", "Balaghat", "Barwani", "Betul", "Bhind", 
    "Bhopal", "Burhanpur", "Chhatarpur", "Chhindwara", "Damoh", "Datia", "Dewas", "Dhar", 
    "Dindori", "Guna", "Gwalior", "Harda", "Hoshangabad", "Indore", "Jabalpur", "Jhabua", 
    "Katni", "Khandwa", "Khargone", "Mandla", "Mandsaur", "Morena", "Narsinghpur", "Neemuch", 
    "Niwari", "Panna", "Raisen", "Rajgarh", "Ratlam", "Rewa", "Sagar", "Satna", "Sehore", 
    "Seoni", "Shahdol", "Shajapur", "Sheopur", "Shivpuri", "Sidhi", "Singrauli", "Tikamgarh", 
    "Ujjain", "Umaria", "Vidisha"
]

def generate_bulk_climate_data():
    """Generates daily climate data for all districts from 2017-01-01 to 2026-05-31 in bulk."""
    db = SessionLocal()
    
    # Check if we already have daily climate data
    existing_count = db.query(DailyClimate).count()
    if existing_count > 0:
        print(f"ℹ️ daily_climate table already contains {existing_count} records. Skipping climate simulation.")
        db.close()
        return

    print("📡 Simulating 10-year daily climate history (2017-2026) for all 52 districts...")
    start_date = datetime(2017, 1, 1).date()
    end_date = datetime(2026, 5, 31).date()
    total_days = (end_date - start_date).days + 1
    
    date_list = [start_date + timedelta(days=x) for x in range(total_days)]
    
    records = []
    # Seed random number generator for reproducibility
    np.random.seed(42)
    
    print(f"📊 Generating {len(MP_DISTRICTS) * total_days} daily records...")
    
    for district in MP_DISTRICTS:
        # We can pre-generate numpy arrays for speed
        tmax_base = np.random.uniform(28.0, 42.0, total_days)
        sm_base = np.random.uniform(0.04, 0.38, total_days)
        precip_base = np.random.exponential(scale=4.0, size=total_days)
        precip_base[precip_base < 1.5] = 0.0
        ndvi_base = np.random.uniform(0.15, 0.75, total_days)
        
        for idx, d in enumerate(date_list):
            # Seasonal adjustments (temperature higher in summer, SM higher in monsoon)
            month = d.month
            t_adj = 0.0
            sm_adj = 0.0
            
            # April-June is peak summer
            if month in [4, 5, 6]:
                t_adj = 3.0
                sm_adj = -0.05
            # July-September is monsoon
            elif month in [7, 8, 9]:
                t_adj = -2.0
                sm_adj = 0.12
            # December-January is winter
            elif month in [12, 1]:
                t_adj = -5.0
                sm_adj = -0.02
                
            records.append({
                "date": d,
                "district": district,
                "tmax": float(np.clip(tmax_base[idx] + t_adj, 15.0, 48.0)),
                "soil_moisture": float(np.clip(sm_base[idx] + sm_adj, 0.01, 0.50)),
                "precipitation": float(precip_base[idx] * (2.0 if month in [7, 8, 9] else 0.2)),
                "ndvi": float(np.clip(ndvi_base[idx] + (0.15 if month in [8, 9, 10] else -0.05), 0.05, 0.90))
            })
            
    # Bulk insert for efficiency
    print("💾 Bulk inserting climate records to database...")
    db.execute(insert(DailyClimate), records)
    db.commit()
    print(f"✅ Synced {db.query(DailyClimate).count()} climate records successfully.")
    db.close()

def main():
    print("🛠️ Beginning Full CSI-DiCRA Data Initialization Pipeline...")
    
    # 1. Init Database schema
    init_db()
    
    # 2. Sync PMFBY Yield History (Mock data)
    sync_yield_history()
    
    # 3. Bulk Generate Daily Climate Data
    generate_bulk_climate_data()
    
    # 4. Calculate Season-Aware Baselines
    db = SessionLocal()
    calculate_district_baselines(db)
    db.close()
    
    # 5. Train XGBoost models for all 6 crops
    train_all_models()
    
    # 6. Generate CMIP6 Projections and Risk Predictions
    generate_forecasts()
    
    print("\n🎉 CSI-DiCRA pipeline initialization complete! Database is fully populated.")

if __name__ == "__main__":
    main()
