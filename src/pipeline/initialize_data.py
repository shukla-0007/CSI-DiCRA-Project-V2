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
from src.pipeline.dicra_fetcher import fetch_dicra_data
from src.pipeline.imd_spi_calculator import fetch_chirps_precipitation

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
    """Syncs daily climate variables from NASA POWER and Sentinel-10m NDVI in bulk from 2017-01-01 to 2026-05-31."""
    db = SessionLocal()
    
    # Check if we already have daily climate data
    existing_count = db.query(DailyClimate).count()
    if existing_count > 0:
        print(f"ℹ️ daily_climate table already contains {existing_count} records. Skipping API synchronization.")
        db.close()
        return

    print("📡 Syncing 10-year historical climate data (2017-2026) using NASA POWER API...")
    start_date = "2017-01-01"
    end_date = "2026-05-31"
    
    records = []
    
    for district in MP_DISTRICTS:
        print(f"🔄 Syncing weather variables for {district}...")
        
        # 1. Fetch temperature, soil moisture, and NDVI from NASA POWER + Sentinel vector NDVI
        dicra_df = fetch_dicra_data(district, start_date, end_date)
        
        # 2. Fetch daily precipitation (PRECTOTCORR)
        precip_df = fetch_chirps_precipitation(district, start_date, end_date)
        
        # 3. Merge the datasets
        merged_df = pd.merge(dicra_df, precip_df, on=["date", "district"], how="inner")
        
        for _, row in merged_df.iterrows():
            d_val = row["date"].date() if hasattr(row["date"], "date") else row["date"]
            records.append({
                "date": d_val,
                "district": row["district"],
                "tmax": float(row["tmax"]) if row["tmax"] is not None else None,
                "soil_moisture": float(row["soil_moisture"]) if row["soil_moisture"] is not None else None,
                "precipitation": float(row["precipitation"]) if row["precipitation"] is not None else None,
                "ndvi": float(row["ndvi"]) if row["ndvi"] is not None else None
            })
            
        print(f"✅ Prepared {len(merged_df)} daily records for {district}.")
        
    if records:
        print(f"💾 Bulk inserting {len(records)} climate records into daily_climate table...")
        db.execute(insert(DailyClimate), records)
        db.commit()
        print("✅ DailyClimate records synchronized and populated successfully.")
    else:
        print("⚠️ No records were prepared for synchronization.")
        
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
