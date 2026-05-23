# Scientific Report: Empirical Machine Learning Limitations in Climate Change Projections

This document details the scientific and algorithmic explanations for the counter-intuitive behavior observed in the CSI-DiCRA model, where the extreme warming scenario (**SSP5-8.5**) results in lower crop risk predictions (greener advisory maps) than the moderate scenario (**SSP2-4.5**).

---

## 1. Case Study Overview: Chhatarpur Soybean (Kharif 2032)

When querying the dashboard for Soybean risk in Chhatarpur district for the year 2032, we observe:
*   **SSP2-4.5 (Moderate Scenario)**: Shows several districts categorized under higher risk tiers (yellow and orange).
*   **SSP5-8.5 (Worst-Case Scenario)**: Projects the entire state as green (Risk Tier 1 — Normal Season).

### Data Verification Check
By checking the underlying daily climate data in the cache files (`data/processed/cmip6_chhatarpur_SSP*.csv`) during the Soybean flowering window (July 25 to August 31), we verify that the climate models are projecting correct physical parameters:
*   **SSP2-4.5 (EC-Earth3P-HR)**: Daily maximum temperatures hover between **$30.1^\circ\text{C}$ and $33.5^\circ\text{C}$**, with high soil moisture (**$0.37$ to $0.46$**).
*   **SSP5-8.5 (MRI-AGCM3-2-S)**: Daily maximum temperatures are extremely high, hovering between **$35.3^\circ\text{C}$ and $39.9^\circ\text{C}$**, with low soil moisture (**$0.15$ to $0.26$**).

Although the climate data correctly shows that SSP5-8.5 is significantly hotter and drier, our machine learning model assigns a **lower risk** to the SSP5-8.5 scenario. 

---

## 2. Core Algorithmic Drivers of the Risk Inversion

This behavior is caused by three key characteristics of decision-tree-based empirical models (like XGBoost):

### A. The Extrapolation Constraint of Decision Trees
Decision trees partition feature spaces into orthogonal boxes based on threshold splits. Once a leaf node is reached, the model outputs a constant prediction representing the mean target value of the training samples in that leaf.

Because decision trees do not use continuous mathematical functions (like linear regression or neural network activations), they **cannot extrapolate trends** beyond the maximum boundaries of their training data.

```
       Historical Range [26°C - 32°C]            Future Out-of-Bounds [35°C - 40°C]
                  
                 (Split at 31°C)                                    
                  /          \                                       
          [Lower Yield]     [High Yield]  ======================>  [Constant High Yield Output]
           (Cloudy/Wet)     (Sunny/Warm)                            (Extrapolation Limit)
```

If the highest split threshold for maximum temperature in the training dataset is $31.5^\circ\text{C}$, any future temperature input above this value—whether it is $32^\circ\text{C}$, $36^\circ\text{C}$, or $40^\circ\text{C}$—will fall into the exact same leaf node and output the exact same prediction.

### B. Confounding Convective Sunshine in Monsoon (Kharif) Crops
In Madhya Pradesh, Soybeans are cultivated during the Kharif season (monsoon). 
*   **Historically (1997–2023)**: Heavy monsoon cloud cover often restricts solar radiation. Years with lower rainfall and slightly higher maximum temperatures ($31^\circ\text{C}$ to $32.5^\circ\text{C}$) often received more sunshine and experienced fewer waterlogging events, resulting in **higher yields**.
*   **The Model's Interpretation**: The XGBoost algorithm associated higher temperature anomalies during the flowering window with optimal photosynthesis and increased yields.
*   **The Future Projection**: When the extreme MRI-AGCM3-2-S model under SSP5-8.5 projects temperatures up to $39^\circ\text{C}$, the model maps these extreme values directly to the positive "high yield / clear skies" leaf node learned from the historical dataset, ignoring the severe physiological heat stress.

### C. Absence of Physiological Tipping Points in Empirical Data
Biological crops experience sharp physiological tipping points. For Soybean:
*   Temperatures above **$35^\circ\text{C}$** cause pollen sterility, flower drop, and rapid transpiration deficits.
*   These relationships are non-linear and biological.

Because the historical training data (1997-2023) contains almost no occurrences of sustained $38^\circ\text{C}$ temperatures during the monsoon window, the statistical model has **no empirical way to learn these heat stress tipping points**.

---

## 3. Recommended Engineering & Modeling Remediation

To address these limitations in a production-level climate advisory system, we recommend the following approaches:

### Option 1: Physics-Guided ML (Constraint Penalties)
Inject crop-science knowledge directly into the inference pipeline by applying a biological penalty term to the predicted yield when temperatures exceed physiological limits:

```python
def apply_biological_envelope(predicted_yield, mean_tmax, crop):
    # For Soybean, yields decline rapidly when temperatures exceed 35°C
    if crop == "Soybean" and mean_tmax > 35.0:
        heat_stress_factor = max(0.0, 1.0 - 0.15 * (mean_tmax - 35.0))
        return predicted_yield * heat_stress_factor
    return predicted_yield
```

### Option 2: Hybrid Modeling (APSIM/DSSAT Ingestion)
Instead of training XGBoost solely on historical yields, train the model on synthetic yields generated by process-based crop growth engines (such as **APSIM** or **DSSAT**). These engines use physical differential equations to simulate transpiration, nitrogen uptake, and thermal stress, allowing the model to simulate crop behavior under extreme climate pathways.

### Option 3: Scenario Transparency Warnings (UI/UX)
Display a notice in the dashboard frontend under the climate scenario selector:
> ⚠️ **Methodological Note:** XGBoost is trained on historical ranges (1997–2023). Under the extreme warming pathway (SSP5-8.5), projected temperatures exceed historical limits. Due to non-extrapolation limits, risk in the SSP5-8.5 scenario may be underestimated.
