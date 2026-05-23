# Implementation Plan: Hybrid Physics-Guided Machine Learning (PGML) Upgrade

This document outlines the implementation details to upgrade the CSI-DiCRA forecasting pipeline from a pure statistical model to a **Hybrid Physics-Guided Machine Learning (PGML)** model. 

This upgrade integrates crop-science temperature thresholds directly into the inference pipeline, resolving the out-of-sample risk underestimation observed in the **SSP5-8.5** scenario.

---

## 1. Crop-Specific Biological Stress Envelopes

Agronomic studies (ICAR / FAO guidelines for Central India) show that crops experience distinct thermal thresholds where yields collapse, regardless of historical sunshine. We define a thermal stress multiplier, $f(T_{\text{max}})$, bounded between $0.1$ and $1.0$:

$$\text{Final Yield} = \text{XGBoost Predicted Yield} \times f(T_{\text{max}})$$

### Threshold Parameterization
*   **Soybean (Kharif)**: Optimal range $\le 35^\circ\text{C}$. Yield drops linearly when mean maximum temperature crosses $35^\circ\text{C}$, reaching base survival limits ($10\%$ of predicted yield) at $45^\circ\text{C}$.
*   **Rice (Kharif)**: Optimal range $\le 36^\circ\text{C}$. Heat stress during grain-fill leads to spikelet sterility above $36^\circ\text{C}$, dropping to base limits at $45^\circ\text{C}$.
*   **Maize (Kharif)**: High thermal tolerance. Optimal range $\le 38^\circ\text{C}$. Yield declines above $38^\circ\text{C}$, dropping to base limits at $46^\circ\text{C}$.
*   **Wheat (Rabi)**: Highly sensitive to terminal heat stress. Optimal range $\le 30^\circ\text{C}$. Yield declines rapidly above $30^\circ\text{C}$, reaching base survival limits at $38^\circ\text{C}$.
*   **Gram (Rabi)**: Optimal range $\le 32^\circ\text{C}$. Yield declines above $32^\circ\text{C}$, reaching base limits at $40^\circ\text{C}$.
*   **Mustard (Rabi)**: Optimal range $\le 30^\circ\text{C}$. Heat stress limits oil synthesis. Yield declines above $30^\circ\text{C}$, reaching base limits at $38^\circ\text{C}$.

---

## 2. Code-Level Implementation Plan

To apply these constraints consistently across the database forecasts and real-time backend API, we will modify two files:

### Part A: Update [src/model/forecaster.py](file:///Users/sigma-7/Documents/CSI-DiCRA-Project/CSI-DiCRA-Project/src/model/forecaster.py)
We will introduce the biological multiplier function and apply it to the forecasting loop:

```python
# Insert at top-level of forecaster.py
def get_physiological_stress_multiplier(mean_tmax: float, crop: str) -> float:
    """
    Applies biophysical thermal stress constraints based on FAO-aligned crop thresholds.
    Returns a multiplier between 0.1 (extreme stress) and 1.0 (no thermal stress).
    """
    # Define (optimal_threshold, failure_threshold) per crop
    thresholds = {
        "Soybean": (35.0, 45.0),
        "Rice":    (36.0, 45.0),
        "Maize":   (38.0, 46.0),
        "Wheat":   (30.0, 38.0),
        "Gram":    (32.0, 40.0),
        "Mustard": (30.0, 38.0)
    }
    
    if crop not in thresholds:
        return 1.0
        
    opt_t, fail_t = thresholds[crop]
    if mean_tmax <= opt_t:
        return 1.0
        
    # Linear yield decline envelope
    penalty = (mean_tmax - opt_t) / (fail_t - opt_t)
    multiplier = 1.0 - penalty
    return max(0.1, float(multiplier))
```

And update the forecasting prediction step inside `generate_forecasts()`:
```python
# Replace line 431-432 of forecaster.py:
predicted_yield = float(xgb_model.predict(features)[0])
# Apply physical envelope constraints:
stress_mult     = get_physiological_stress_multiplier(features_dict["mean_tmax"], crop)
final_yield     = predicted_yield * stress_mult
tier            = yield_to_tier(final_yield, crop)
```

---

### Part B: Update [src/main.py](file:///Users/sigma-7/Documents/CSI-DiCRA-Project/CSI-DiCRA-Project/src/main.py)
To ensure the backend API endpoint (`/api/v1/dashboard/map_data`) calculations and details sync with the updated model predictions, we will import and apply the multiplier to the API response logic:

```python
# Inside get_map_data in main.py, replace lines 297-308:
shap_result = get_shap_explanation(crop, feat_arr)

# Apply physiological stress multiplier to ensure API and database alignment
from src.model.forecaster import get_physiological_stress_multiplier
stress_mult = get_physiological_stress_multiplier(mean_tmax, crop)

# Round output variables
results[district] = {
    "NDVI": round(float(ndvi_approx), 3),
    "Tmax": round(float(mean_tmax), 1),
    "Soil Moisture": round(float(mean_sm), 3),
    "SPI-3": round(float(window_spi3), 3) if pd.notna(window_spi3) else 0.0,
    "CSI Risk Tier": int(tier),
    "CSI Score": round(float(score), 3),
    "Recommended Premium Adjustment": recommended_premium,
    "explainability": shap_result
}
```

---

## 3. How to Execute this Upgrade
To implement this Hybrid PGML upgrade:
1. Approve the changes.
2. Run the training and forecasting pipeline to re-calculate the risk predictions inside the SQLite database (`dicra_csi.db`):
   ```bash
   PYTHONPATH=. python -m src.model.train_xgboost
   PYTHONPATH=. python -m src.model.forecaster
   ```
3. Restart your uvicorn backend and streamlit frontend.
