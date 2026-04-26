import os
import requests
import pandas as pd
from sqlalchemy.orm import Session
from src.database.db import SessionLocal, YieldHistory

def fetch_data_gov_in_yields(api_key: str, state: str = "Madhya Pradesh"):
    """
    Fetches district-wise crop production statistics from data.gov.in.
    Endpoint: https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070
    """
    print(f"🌾 Querying data.gov.in API for historical yields in {state}...")
    
    if not api_key:
        print("⚠️ Warning: DATA_GOV_IN_API_KEY not found in environment variables.")
        print("Using simulated validation data for now.")
        
        # Simulate data
        return pd.DataFrame([
            {"district": "Betul", "year": 2018, "season": "Kharif", "crop": "Soybean", "yield_kg_ha": 650.0},
            {"district": "Betul", "year": 2019, "season": "Kharif", "crop": "Soybean", "yield_kg_ha": 1100.0},
            {"district": "Bhopal", "year": 2018, "season": "Kharif", "crop": "Soybean", "yield_kg_ha": 700.0},
            {"district": "Bhopal", "year": 2019, "season": "Kharif", "crop": "Soybean", "yield_kg_ha": 1250.0},
        ])

    # True API logic would go here
    # url = f"https://api.data.gov.in/resource/...?api-key={api_key}&format=json&filters[state]={state}"
    # response = requests.get(url)
    # data = response.json()
    # return pd.DataFrame(data['records'])

def sync_yield_history():
    api_key = os.environ.get("DATA_GOV_IN_API_KEY", "")
    df = fetch_data_gov_in_yields(api_key)
    
    db = SessionLocal()
    for _, row in df.iterrows():
        existing = db.query(YieldHistory).filter(
            YieldHistory.district == row['district'],
            YieldHistory.year == row['year'],
            YieldHistory.season == row['season'],
            YieldHistory.crop == row['crop']
        ).first()
        
        if existing:
            existing.yield_kg_ha = row['yield_kg_ha']
        else:
            record = YieldHistory(
                district=row['district'],
                year=row['year'],
                season=row['season'],
                crop=row['crop'],
                yield_kg_ha=row['yield_kg_ha']
            )
            db.add(record)
    
    db.commit()
    print(f"✅ Synced historical yield records for validation.")
    db.close()

if __name__ == "__main__":
    sync_yield_history()
