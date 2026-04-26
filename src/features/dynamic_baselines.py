import pandas as pd
from datetime import date
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.database.db import SessionLocal, DailyClimate, DistrictBaselines

def calculate_district_baselines(db: Session):
    """
    Calculates historical baselines for each district based on the entire
    DailyClimate dataset (or a rolling 15-year window if dataset is large).
    Calculates 95th percentile Tmax and 5th percentile Soil Moisture.
    """
    print("🔄 Calculating Dynamic Baselines (95th Heat, 5th Drought)...")
    
    # Get distinct districts
    districts = db.query(DailyClimate.district).distinct().all()
    districts = [d[0] for d in districts]
    
    for district in districts:
        records = db.query(DailyClimate).filter(DailyClimate.district == district).all()
        if not records:
            continue
            
        df = pd.DataFrame([{
            'tmax': r.tmax,
            'soil_moisture': r.soil_moisture
        } for r in records])
        
        # Calculate percentiles
        tmax_95th = df['tmax'].quantile(0.95)
        sm_5th = df['soil_moisture'].quantile(0.05)
        
        # Upsert baseline
        existing = db.query(DistrictBaselines).filter(DistrictBaselines.district == district).first()
        if existing:
            existing.tmax_95th = tmax_95th
            existing.sm_5th = sm_5th
            existing.calculated_on = date.today()
        else:
            record = DistrictBaselines(
                district=district,
                tmax_95th=tmax_95th,
                sm_5th=sm_5th,
                calculated_on=date.today()
            )
            db.add(record)
            
        print(f"📍 {district}: Tmax 95th={tmax_95th:.1f}°C, SM 5th={sm_5th:.3f}")
        
    db.commit()
    print("✅ Baseline calculation complete.")

if __name__ == "__main__":
    db = SessionLocal()
    calculate_district_baselines(db)
    db.close()





