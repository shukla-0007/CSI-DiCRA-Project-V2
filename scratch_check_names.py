import json

with open("data/mp_boundaries.json", "r") as f:
    data = json.load(f)

names = [feature["properties"]["district_name"] for feature in data["features"]]
print("Number of features:", len(names))
print("Names in GeoJSON:", sorted(names))
