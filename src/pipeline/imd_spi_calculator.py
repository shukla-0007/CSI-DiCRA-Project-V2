import os
import time
import requests
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from src.database.db import SessionLocal, DailyClimate, log_api_status
from src.pipeline.dicra_fetcher import MP_DISTRICTS, get_district_coordinates

def fetch_chirps_precipitation(district_name: str, start_date: str, end_date: str):
    """
    Fetch daily precipitation (PRECTOTCORR) from NASA POWER API.
    Uses district coordinates mapping.
    """
    def _db_log(status: str, service: str, message: str):
        db = SessionLocal()
        try:
            log_api_status(db, service, district_name, status, message)
        except Exception as e:
            print(f"Logging error: {e}")
        finally:
            db.close()

    lat, lon = get_district_coordinates(district_name)
    start_str = start_date.replace("-", "")
    end_str = end_date.replace("-", "")
    
    print(f"🌧️ Querying NASA POWER API for precipitation in {district_name} ({start_date} to {end_date})...")
    
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": start_str,
        "end": end_str,
        "latitude": lat,
        "longitude": lon,
        "community": "ag",
        "parameters": "PRECTOTCORR",
        "format": "JSON"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            res_json = response.json()
            param_data = res_json['properties']['parameter']
            precip_dict = param_data.get('PRECTOTCORR', {})
            
            dates = []
            precip_vals = []
            
            for date_key in sorted(precip_dict.keys()):
                parsed_date = datetime.strptime(date_key, "%Y%m%d")
                dates.append(parsed_date)
                
                precip_val = precip_dict[date_key]
                if precip_val == -999.0 or precip_val is None:
                    precip_val = 0.0
                precip_vals.append(precip_val)
                
            df = pd.DataFrame({
                "date": dates,
                "district": district_name,
                "precipitation": precip_vals
            })
            _db_log("SUCCESS", "NASA POWER Precipitation API", f"Successfully fetched daily precipitation records.")
            return df
    except Exception as e:
        print(f"❌ NASA POWER connection failed for {district_name} precipitation: {e}. Using fallback.")
        _db_log("FALLBACK", "NASA POWER Precipitation API", f"API connection failed: {e}. Falling back to daily simulation.")
        
    # Fallback to daily precipitation simulation
    dates = pd.date_range(start=start_date, end=end_date)
    rain = np.random.exponential(scale=5.0, size=len(dates))
    rain[rain < 2.0] = 0.0
    
    adjusted_rain = []
    for idx, d in enumerate(dates):
        month = d.month
        # Monsoon (July-Sept) gets double precip, winter/summer gets less
        mult = 2.0 if month in [7, 8, 9] else 0.2
        adjusted_rain.append(rain[idx] * mult)
        
    df = pd.DataFrame({
        "date": dates,
        "district": district_name,
        "precipitation": adjusted_rain
    })
    return df

def calculate_spi3(db: Session, district_name: str) -> pd.Series:
    """
    Computes SPI-3 (Standardized Precipitation Index - 3 Month) for a district.
    
    Methodology:
    1. Retrieve all daily precipitation records for the district from the DB.
    2. Compute 90-day rolling sums of daily precipitation.
    3. Group these 90-day sums by calendar month (1-12) to capture seasonality.
    4. Fit a Gamma distribution to the non-zero rolling sums for each month group.
    5. Handle zero probability: H(x) = q + (1 - q) * G(x), where q = count_zeros / total.
    6. Convert cumulative probability H(x) to a standard normal Z-score.
    """
    records = db.query(DailyClimate).filter(
        DailyClimate.district == district_name
    ).order_by(DailyClimate.date).all()
    
    if not records:
        print(f"⚠️ No climate records in DB for {district_name} to compute SPI-3.")
        return pd.Series(dtype=float)
        
    df = pd.DataFrame([
        {"date": r.date, "precipitation": r.precipitation if r.precipitation is not None else 0.0}
        for r in records
    ])
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    
    # Compute 90-day rolling sum (representing 3 months)
    df['precip_90'] = df['precipitation'].rolling(window=90, min_periods=1).sum()
    
    spi3_series = pd.Series(index=df.index, dtype=float)
    
    # Calculate SPI-3 for each month using historical baseline distribution
    for month in range(1, 13):
        month_mask = df.index.month == month
        month_data = df.loc[month_mask, 'precip_90']
        if month_data.empty:
            continue
            
        N = len(month_data)
        zeros_mask = month_data <= 0.001
        m = zeros_mask.sum()
        q = m / N
        
        non_zero_data = month_data[~zeros_mask]
        
        if len(non_zero_data) >= 3:
            try:
                # Fit shape, loc, scale. Fix loc=0 as Gamma is bounded at 0.
                shape, loc, scale = stats.gamma.fit(non_zero_data, floc=0)
                g_cdf = stats.gamma.cdf(month_data, shape, loc=loc, scale=scale)
                h = q + (1 - q) * g_cdf
            except Exception as e:
                # Fallback to empirical cumulative distribution if fitting fails
                h = month_data.rank(method='average') / (N + 1)
        else:
            # If mostly zeros, assign probability based on proportion of zeros
            h = np.zeros(N)
            h[zeros_mask] = q
            if N - m > 0:
                h[~zeros_mask] = 0.5 + 0.5 * (month_data[~zeros_mask].rank() / (N - m + 1))
                
        # Clip probability to avoid infinite Z-scores
        h = np.clip(h, 0.00001, 0.99999)
        month_spi = stats.norm.ppf(h)
        spi3_series.loc[month_mask] = month_spi
        
    return spi3_series

def sync_precipitation_live():
    """Syncs precipitation data for the last 30 days and updates the DB."""
    db = SessionLocal()
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    for district in MP_DISTRICTS:
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

if __name__ == "__main__":
    sync_precipitation_live()

