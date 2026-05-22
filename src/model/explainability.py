import os
import joblib
import numpy as np
import shap

# Cache explainers globally to avoid reloading models and re-initializing explainers on every API call.
_EXPLAINERS = {}

def get_shap_explanation(crop: str, features: np.ndarray) -> dict:
    """
    Computes SHAP values for a single prediction feature vector.
    
    Parameters:
    - crop (str): The name of the crop (e.g., 'Soybean', 'Wheat').
    - features (np.ndarray): Shape (1, 6) matching the trained model's feature vector:
      [mean_tmax, mean_sm, mean_ndvi, tot_precip, tmax_anomaly, sm_anomaly]

    Returns:
    - dict: A dictionary with 'base_value' and 'contributions' mapping each feature to its SHAP value.
      A negative SHAP value reduces predicted yield (increases crop stress/risk).
      A positive SHAP value increases predicted yield (reduces crop stress/risk).
    """
    crop_lower = crop.lower()
    model_path = f"models/xgboost_{crop_lower}_model.pkl"
    
    if not os.path.exists(model_path):
        # Return empty contribution mapping if model not found
        return {
            "base_value": 0.0,
            "contributions": {}
        }

    # Load/Cache model and explainer
    if crop_lower not in _EXPLAINERS:
        try:
            model = joblib.load(model_path)
            explainer = shap.TreeExplainer(model)
            _EXPLAINERS[crop_lower] = (model, explainer)
        except Exception as e:
            print(f"⚠️ Error loading SHAP explainer for {crop}: {e}")
            return {
                "base_value": 0.0,
                "contributions": {}
            }

    model, explainer = _EXPLAINERS[crop_lower]
    
    try:
        # shap_values shape: (1, 6)
        shap_vals = explainer.shap_values(features)[0]
        
        # In newer versions of SHAP, expected_value might be a list or array
        base_val = explainer.expected_value
        if isinstance(base_val, (np.ndarray, list)):
            base_val = float(base_val[0])
        else:
            base_val = float(base_val)

        feature_names = [
            "Mean Temperature",
            "Soil Moisture",
            "NDVI Vegetation Index",
            "Total Precipitation",
            "Temperature Anomaly",
            "Soil Moisture Anomaly"
        ]

        contributions = {
            name: round(float(val), 2)
            for name, val in zip(feature_names, shap_vals)
        }

        return {
            "base_value": round(base_val, 2),
            "contributions": contributions
        }
    except Exception as e:
        print(f"⚠️ SHAP calculation failed for {crop}: {e}")
        return {
            "base_value": 0.0,
            "contributions": {}
        }
