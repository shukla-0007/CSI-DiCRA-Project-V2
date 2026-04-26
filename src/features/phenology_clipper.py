import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from src.database.db import SessionLocal, DailyClimate

# FAO Crop Calendars for Madhya Pradesh
CROP_CALENDARS = {
    "Soybean": {"season": "Kharif", "critical_start": "08-15", "critical_end": "09-15"},
    "Wheat": {"season": "Rabi", "critical_start": "01-15", "critical_end": "02-15"},
    "Cotton": {"season": "Kharif", "critical_start": "08-01", "critical_end": "09-30"}
}

def get_clipped_climate_data(db: Session, district: str, year: int, crop: str):
    """
    Retrieves only the climate records for a specific district and year
    that fall within the critical flowering/growth window for the given crop.
    """
    if crop not in CROP_CALENDARS:
        raise ValueError(f"Crop {crop} not supported.")
        
    cal = CROP_CALENDARS[crop]
    start_str = f"{year}-{cal['critical_start']}"
    end_str = f"{year}-{cal['critical_end']}"
    
    start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
    
    records = db.query(DailyClimate).filter(
        DailyClimate.district == district,
        DailyClimate.date >= start_date,
        DailyClimate.date <= end_date
    ).all()
    
    return records

if __name__ == "__main__":
    print("Testing Phenology Clipper...")
    db = SessionLocal()
    # Assuming some data exists from 2026 for Betul
    clipped = get_clipped_climate_data(db, "Betul", 2026, "Soybean")
    print(f"Retrieved {len(clipped)} days of critical phenological data for Soybean.")
    db.close()
