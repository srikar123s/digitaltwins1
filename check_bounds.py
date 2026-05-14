import json

for region in ['uttarakhand', 'assam', 'western_ghats']:
    try:
        with open(f'outputs/{region}_historical_summary.geojson') as f:
            d = json.load(f)
        lats, lons = [], []
        for feat in d['features']:
            coords = feat['geometry']['coordinates'][0]
            lats += [c[1] for c in coords]
            lons += [c[0] for c in coords]
        print(f"{region}:")
        print(f"  lat: {min(lats):.3f} to {max(lats):.3f}")
        print(f"  lon: {min(lons):.3f} to {max(lons):.3f}")
        print(f"  cells: {len(d['features'])}")
    except Exception as e:
        print(f"{region}: ERROR - {e}")
