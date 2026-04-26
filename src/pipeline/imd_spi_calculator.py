import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from src.database.db import SessionLocal, DailyClimate

def fetch_chirps_precipitation(district_name: str, start_date: str, end_date: str):
    """
    Mock function representing fetching daily precipitation from CHIRPS 
    (Climate Hazards Group InfraRed Precipitation with Station data).
    CHIRPS has a public REST API / THREDDS server.
    """
    print(f"🌧️ Fetching CHIRPS Precipitation for {district_name} ({start_date} to {end_date})...")
    time.sleep(0.5)
    
    dates = pd.date_range(start=start_date, end=end_date)
    # Simulate rainfall data (many 0s, some rain events)
    rain = np.random.exponential(scale=5.0, size=len(dates))
    rain[rain < 2.0] = 0.0 # Make small values 0 to simulate dry days
    
    data = {
        "date": dates,
        "district": district_name,
        "precipitation": rain
    }
    return pd.DataFrame(data)

def sync_precipitation_live():
    """Syncs precipitation data and updates the DB."""
    db = SessionLocal()
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    districts = [
        "Agar Malwa", "Alirajpur", "Anuppur", "Ashoknagar", "Balaghat", "Barwani", "Betul", "Bhind", 
        "Bhopal", "Burhanpur", "Chhatarpur", "Chhindwara", "Damoh", "Datia", "Dewas", "Dhar", 
        "Dindori", "Guna", "Gwalior", "Harda", "Hoshangabad", "Indore", "Jabalpur", "Jhabua", 
        "Katni", "Khandwa", "Khargone", "Mandla", "Mandsaur", "Morena", "Narsinghpur", "Neemuch", 
        "Niwari", "Panna", "Raisen", "Rajgarh", "Ratlam", "Rewa", "Sagar", "Satna", "Sehore", 
        "Seoni", "Shahdol", "Shajapur", "Sheopur", "Shivpuri", "Sidhi", "Singrauli", "Tikamgarh", 
        "Ujjain", "Umaria", "Vidisha"
    ]
    
    for district in districts:
        df = fetch_chirps_precipitation(district, start_str, end_str)
        
        for _, row in df.iterrows():
            existing = db.query(DailyClimate).filter(
                DailyClimate.date == row['date'].date(),
                DailyClimate.district == row['district']
            ).first()
            
            if existing:
                existing.precipitation = row['precipitation']
            else:
                record = DailyClimate(
                    date=row['date'].date(),
                    district=row['district'],
                    precipitation=row['precipitation']
                )
                db.add(record)
                
        db.commit()
        print(f"✅ Synced {len(df)} precipitation records for {district}.")
        
    db.close()

# Note: SPI calculation (Standardized Precipitation Index) requires 
# fitting a gamma distribution to historical monthly totals.
# For daily real-time updates, we just sync the daily precipitation.
# A separate batch job would compute SPI-3 and SPI-6 at the end of each month.

if __name__ == "__main__":
    sync_precipitation_live()
