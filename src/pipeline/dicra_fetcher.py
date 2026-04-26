import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from src.database.db import SessionLocal, DailyClimate

# We assume a list of target districts for Madhya Pradesh
MP_DISTRICTS = [
    "Agar Malwa", "Alirajpur", "Anuppur", "Ashoknagar", "Balaghat", "Barwani", "Betul", "Bhind", 
    "Bhopal", "Burhanpur", "Chhatarpur", "Chhindwara", "Damoh", "Datia", "Dewas", "Dhar", 
    "Dindori", "Guna", "Gwalior", "Harda", "Hoshangabad", "Indore", "Jabalpur", "Jhabua", 
    "Katni", "Khandwa", "Khargone", "Mandla", "Mandsaur", "Morena", "Narsinghpur", "Neemuch", 
    "Niwari", "Panna", "Raisen", "Rajgarh", "Ratlam", "Rewa", "Sagar", "Satna", "Sehore", 
    "Seoni", "Shahdol", "Shajapur", "Sheopur", "Shivpuri", "Sidhi", "Singrauli", "Tikamgarh", 
    "Ujjain", "Umaria", "Vidisha"
]

def fetch_dicra_data(district_name: str, start_date: str, end_date: str):
    """
    Mock function to represent fetching data from DiCRA's REST APIs.
    In a true live scenario, this would query the UNDP DiCRA Geoserver or GEE app
    for zonal statistics (mean Tmax, mean Root Zone Soil Moisture, mean NDVI) 
    clipped to the district geojson.
    """
    print(f"📡 Fetching DiCRA Tmax & Soil Moisture for {district_name} ({start_date} to {end_date})...")
    
    # Simulate an API delay
    time.sleep(0.5)
    
    # Generate a date range
    dates = pd.date_range(start=start_date, end=end_date)
    
    # Simulate data that might be returned
    # Tmax ~30-45C, Soil Moisture ~0.02-0.4, NDVI ~0.1-0.8
    import numpy as np
    data = {
        "date": dates,
        "district": district_name,
        "tmax": np.random.uniform(30.0, 45.0, len(dates)),
        "soil_moisture": np.random.uniform(0.02, 0.40, len(dates)),
        "ndvi": np.random.uniform(0.1, 0.8, len(dates))
    }
    
    return pd.DataFrame(data)

def update_database(db: Session, df: pd.DataFrame):
    """Upsert the fetched data into the SQLite database."""
    records = []
    for _, row in df.iterrows():
        # Check if record exists
        existing = db.query(DailyClimate).filter(
            DailyClimate.date == row['date'].date(),
            DailyClimate.district == row['district']
        ).first()
        
        if existing:
            existing.tmax = row['tmax']
            existing.soil_moisture = row['soil_moisture']
            existing.ndvi = row['ndvi']
        else:
            record = DailyClimate(
                date=row['date'].date(),
                district=row['district'],
                tmax=row['tmax'],
                soil_moisture=row['soil_moisture'],
                ndvi=row['ndvi']
            )
            db.add(record)
            
    db.commit()
    print(f"✅ Synced {len(df)} records to database.")

def sync_dicra_live():
    """
    Main function to sync missing DiCRA data up to the current date.
    """
    db = SessionLocal()
    
    # Determine the date range to fetch. We'll fetch the last 30 days for this example.
    # In production, you query the DB to find the 'max(date)' for each district and fetch from there.
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    for district in MP_DISTRICTS:
        df = fetch_dicra_data(district, start_str, end_str)
        update_database(db, df)
        
    db.close()

if __name__ == "__main__":
    sync_dicra_live()
