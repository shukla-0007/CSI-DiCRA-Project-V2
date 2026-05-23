import json
import copy
import pandas as pd
import plotly.graph_objects as go

# 1. Load GeoJSON
with open("data/mp_boundaries.json", "r") as f:
    geojson = json.load(f)

# Rename properties to match DB district names
rename_map = {
    "AgarMalwa": "Agar Malwa",
    "EastNimar": "Khandwa",
    "WestNimar": "Khargone",
    "Narsimhapur": "Narsinghpur"
}

tikamgarh_feature = None
for feature in geojson["features"]:
    name = feature["properties"]["district_name"]
    if name in rename_map:
        feature["properties"]["district_name"] = rename_map[name]
    if name == "Tikamgarh":
        tikamgarh_feature = feature

# Duplicate Tikamgarh for Niwari
if tikamgarh_feature:
    niwari_feature = copy.deepcopy(tikamgarh_feature)
    niwari_feature["properties"]["district_name"] = "Niwari"
    geojson["features"].append(niwari_feature)

# 2. Create sample dummy data for all 52 districts
districts = [f["properties"]["district_name"] for f in geojson["features"]]
df = pd.DataFrame({
    "district": districts,
    "value": [idx * 1.5 for idx, _ in enumerate(districts)]
})

# 3. Create figure
fig = go.Figure(go.Choroplethmap(
    geojson=geojson,
    locations=df["district"],
    z=df["value"],
    featureidkey="properties.district_name",
    colorscale="Viridis",
    marker_opacity=0.6,
    marker_line_width=1,
    marker_line_color="#21262d",
))

# Bhopal coordinates centroid
fig.update_layout(
    map=dict(
        style="carto-darkmatter",
        center=dict(lat=23.25, lon=77.4),
        zoom=5.5
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
)

print("Plotly figure built successfully!")
