import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def test():
    conn = sqlite3.connect('dicra_csi.db')
    
    # Let's inspect soybean first
    crop = 'Soybean'
    
    # Load baselines
    baselines = pd.read_sql_query('SELECT * FROM district_baselines WHERE crop=?', conn, params=(crop,))
    baselines_dict = baselines.set_index('district').to_dict('index')
    
    # Load yields
    yields = pd.read_sql_query('SELECT * FROM yield_history WHERE crop=?', conn, params=(crop,))
    
    # Load all climate data
    climate = pd.read_sql_query('SELECT * FROM daily_climate', conn)
    climate['date'] = pd.to_datetime(climate['date'])
    
    # Let's extract features for soybean (July 25 to Aug 31)
    records = []
    for _, row in yields.iterrows():
        dist = row['district']
        yr = row['year']
        y_val = row['yield_kg_ha']
        
        if dist not in baselines_dict:
            continue
        base = baselines_dict[dist]
        
        # Filter climate to critical window
        mask = (climate['district'] == dist) & (climate['date'].dt.year == yr) & (climate['date'].dt.month == 8) # Aug approx
        # or exactly:
        w_start = pd.Timestamp(f"{yr}-07-25")
        w_end = pd.Timestamp(f"{yr}-08-31")
        w_clim = climate[(climate['district'] == dist) & (climate['date'] >= w_start) & (climate['date'] <= w_end)]
        
        if w_clim.empty:
            continue
            
        mean_tmax = w_clim['tmax'].mean()
        mean_sm = w_clim['soil_moisture'].mean()
        mean_ndvi = w_clim['ndvi'].mean()
        tot_precip = w_clim['precipitation'].sum()
        
        comp_ev = sum(
            1 for _, r in w_clim.iterrows()
            if pd.notna(r['tmax']) and pd.notna(r['soil_moisture'])
            and r['tmax'] > base['tmax_95th']
            and r['soil_moisture'] < base['sm_5th']
        )
        
        # Anomaly based features
        tmax_anomaly = mean_tmax - base['tmax_95th']
        sm_anomaly = mean_sm - base['sm_5th']
        
        records.append({
            'district': dist,
            'year': yr,
            'mean_tmax': mean_tmax,
            'mean_sm': mean_sm,
            'mean_ndvi': mean_ndvi,
            'tot_precip': tot_precip,
            'comp_ev': comp_ev,
            'tmax_anomaly': tmax_anomaly,
            'sm_anomaly': sm_anomaly,
            'yield': y_val
        })
        
    df = pd.DataFrame(records)
    print("Correlations with yield:")
    print(df[['mean_tmax', 'mean_sm', 'mean_ndvi', 'tot_precip', 'comp_ev', 'tmax_anomaly', 'sm_anomaly', 'yield']].corr()['yield'])
    
    # Train test split
    X = df[['mean_tmax', 'mean_sm', 'mean_ndvi', 'tot_precip', 'comp_ev', 'tmax_anomaly', 'sm_anomaly']]
    y = df['yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"\nModel RMSE: {rmse:.2f}")
    
    # Feature importances
    for col, imp in zip(X.columns, model.feature_importances_):
        print(f"  {col}: {imp:.4f}")
        
    # Check predictions for high/low features
    print("\nPredicting on test set samples vs actual:")
    res = pd.DataFrame({'Actual': y_test, 'Pred': preds})
    print(res.head(10))
    
    conn.close()

if __name__ == '__main__':
    test()
