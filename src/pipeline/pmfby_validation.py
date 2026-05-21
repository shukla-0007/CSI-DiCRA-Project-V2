import os
import requests
import pandas as pd
from sqlalchemy.orm import Session
from src.database.db import SessionLocal, YieldHistory

# ── Mock yield data — used when DATA_GOV_IN_API_KEY is absent ──────────────────
# Values are representative of real PMFBY CCE records for Madhya Pradesh.
# Two districts × two years per crop gives the trainer enough rows to run
# end-to-end until real API data is connected.
MOCK_YIELDS = [
    # Soybean (Kharif) — typical range 900-1100 kg/ha
    {"district": "Betul",  "year": 2018, "season": "Kharif", "crop": "Soybean", "yield_kg_ha":  650.0},
    {"district": "Betul",  "year": 2019, "season": "Kharif", "crop": "Soybean", "yield_kg_ha": 1100.0},
    {"district": "Bhopal", "year": 2018, "season": "Kharif", "crop": "Soybean", "yield_kg_ha":  700.0},
    {"district": "Bhopal", "year": 2019, "season": "Kharif", "crop": "Soybean", "yield_kg_ha": 1050.0},
    # Rice (Kharif) — typical range 900-1500 kg/ha
    {"district": "Balaghat",  "year": 2018, "season": "Kharif", "crop": "Rice", "yield_kg_ha":  820.0},
    {"district": "Balaghat",  "year": 2019, "season": "Kharif", "crop": "Rice", "yield_kg_ha": 1380.0},
    {"district": "Chhindwara","year": 2018, "season": "Kharif", "crop": "Rice", "yield_kg_ha":  950.0},
    {"district": "Chhindwara","year": 2019, "season": "Kharif", "crop": "Rice", "yield_kg_ha": 1200.0},
    # Maize (Kharif) — typical range 1300-3200 kg/ha
    {"district": "Chhatarpur","year": 2018, "season": "Kharif", "crop": "Maize", "yield_kg_ha": 1100.0},
    {"district": "Chhatarpur","year": 2019, "season": "Kharif", "crop": "Maize", "yield_kg_ha": 2200.0},
    {"district": "Sagar",     "year": 2018, "season": "Kharif", "crop": "Maize", "yield_kg_ha": 1350.0},
    {"district": "Sagar",     "year": 2019, "season": "Kharif", "crop": "Maize", "yield_kg_ha": 2500.0},
    # Wheat (Rabi) — typical range 1600-2600 kg/ha
    {"district": "Hoshangabad","year": 2018, "season": "Rabi", "crop": "Wheat", "yield_kg_ha": 1200.0},
    {"district": "Hoshangabad","year": 2019, "season": "Rabi", "crop": "Wheat", "yield_kg_ha": 2400.0},
    {"district": "Vidisha",    "year": 2018, "season": "Rabi", "crop": "Wheat", "yield_kg_ha": 1400.0},
    {"district": "Vidisha",    "year": 2019, "season": "Rabi", "crop": "Wheat", "yield_kg_ha": 2100.0},
    # Gram / Chickpea (Rabi) — typical range 900-1100 kg/ha
    {"district": "Damoh",  "year": 2018, "season": "Rabi", "crop": "Gram", "yield_kg_ha":  480.0},
    {"district": "Damoh",  "year": 2019, "season": "Rabi", "crop": "Gram", "yield_kg_ha": 1020.0},
    {"district": "Sagar",  "year": 2018, "season": "Rabi", "crop": "Gram", "yield_kg_ha":  550.0},
    {"district": "Sagar",  "year": 2019, "season": "Rabi", "crop": "Gram", "yield_kg_ha":  980.0},
    # Mustard (Rabi) — typical range 1000-1200 kg/ha
    {"district": "Morena", "year": 2018, "season": "Rabi", "crop": "Mustard", "yield_kg_ha":  680.0},
    {"district": "Morena", "year": 2019, "season": "Rabi", "crop": "Mustard", "yield_kg_ha": 1150.0},
    {"district": "Bhind",  "year": 2018, "season": "Rabi", "crop": "Mustard", "yield_kg_ha":  720.0},
    {"district": "Bhind",  "year": 2019, "season": "Rabi", "crop": "Mustard", "yield_kg_ha": 1080.0},
]


def fetch_data_gov_in_yields(api_key: str, state: str = "Madhya Pradesh") -> pd.DataFrame:
    """
    Fetches district-wise crop yield statistics from data.gov.in.
    Endpoint: https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070

    Falls back to MOCK_YIELDS (all 6 crops) if no API key is provided or
    if the API call fails.
    """
    if not api_key:
        print("⚠️  DATA_GOV_IN_API_KEY not set — using mock yield data (all 6 crops).")
        return pd.DataFrame(MOCK_YIELDS)

    print(f"🌾 Querying data.gov.in for historical yields in {state}...")
    url = (
        f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        f"?api-key={api_key}&format=json&filters[state]={state}&limit=5000"
    )
    try:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        if "records" in data and data["records"]:
            records = []
            for r in data["records"]:
                try:
                    prod  = float(r.get("production", 0))
                    area  = float(r.get("area", 1))
                    yield_val = (prod / area) * 1000 if area > 0 else 0
                    records.append({
                        "district":    str(r.get("district_name", "")).title(),
                        "year":        int(r.get("crop_year", 0)),
                        "season":      str(r.get("season", "")).strip(),
                        "crop":        str(r.get("crop", "")).strip(),
                        "yield_kg_ha": yield_val,
                    })
                except Exception:
                    pass
            return pd.DataFrame(records)
    except Exception as e:
        print(f"  API Error: {e}")

    print("  Falling back to mock data.")
    return pd.DataFrame(MOCK_YIELDS)


def sync_yield_history():
    """Fetch yield data and upsert into YieldHistory."""
    api_key = os.environ.get("DATA_GOV_IN_API_KEY", "")
    df      = fetch_data_gov_in_yields(api_key)
    db      = SessionLocal()

    for _, row in df.iterrows():
        existing = db.query(YieldHistory).filter(
            YieldHistory.district == row["district"],
            YieldHistory.year     == row["year"],
            YieldHistory.season   == row["season"],
            YieldHistory.crop     == row["crop"],
        ).first()

        if existing:
            existing.yield_kg_ha = row["yield_kg_ha"]
        else:
            db.add(YieldHistory(
                district    = row["district"],
                year        = row["year"],
                season      = row["season"],
                crop        = row["crop"],
                yield_kg_ha = row["yield_kg_ha"],
            ))

    db.commit()
    print(f"✅ Synced {len(df)} yield records across {df['crop'].nunique()} crops.")
    db.close()


if __name__ == "__main__":
    sync_yield_history()
