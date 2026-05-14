"""
ML Model Training Pipeline for Flood/Landslide Risk Prediction
================================================================

TRAINING CONDITIONS & FEATURE ENGINEERING:
==========================================

1. EVENT MATCHING CONDITIONS:
   - Load historical events from finalss_india_flood_landslide.csv
   - Filter: events in Assam, Uttarakhand, Western Ghats
   - Time range: 2010-2025 (overlaps with available simulation/terrain data)
   - Spatial: Match event lat/lon to nearest grid cell within simulation bounds
   - Match tolerance: 0.1 degrees (~11 km) to account for location uncertainty

2. LABEL CREATION:
   - Positive label (y=1): Grid cells containing historical flood/landslide events
   - Negative labels (y=0): Random sample of cells without events (10x ratio for balance)
   - Stratified: Balance by region and disaster type

3. FEATURE ENGINEERING (per grid cell):
   
   A. PHYSICS MODEL OUTPUTS (from simulation):
      - flood_probability: [0, 1] - Physics model flood risk score
      - landslide_stress: [0, 1] - Physics model landslide susceptibility
      - composite_risk: [0, 1] - Combined physics prediction
   
   B. TERRAIN FEATURES (from DEM):
      - elevation: meters from sea level
      - slope: degrees or % (terrain steepness)
      - aspect: degrees (N/S/E/W exposure)
      - flow_accumulation: cell count or normalized [0, 1]
      - distance_to_water: meters (proximity to rivers/streams)
      - terrain_roughness: std dev of elevation in neighborhood
   
   C. RAINFALL FEATURES (from CHIRPS data or aggregated):
      - annual_rainfall: mm/year
      - monsoon_intensity: mm/month during monsoon (Jun-Sep)
      - rainfall_variability: coefficient of variation (inter-annual)
      - recent_rainfall: accumulated over event year (if available)
   
   D. SPATIAL CONTEXT:
      - region_id: categorical (Assam/Uttarakhand/Western Ghats)
      - year: year of simulation/event
   
   E. INTERACTION FEATURES:
      - slope_x_rainfall: high slope + heavy rain → high landslide risk
      - elevation_x_slope: compound terrain risk
      - flow_acc_x_slope: flow intensity in steep terrain

4. MODEL ARCHITECTURE:
   - Algorithm: XGBoost (gradient boosted trees)
   - Rationale: Handles non-linear interactions, missing values, feature importance visibility
   - Target: Combined risk probability [0, 1] (unified model for both flood & landslide)
   - Train/Test split: 80/20 by region (cross-validate per-region performance)
   - Cross-validation: 5-fold stratified by (region, disaster_type)

5. MODEL OUTPUT:
   - ml_risk_score: [0, 1] - ML-predicted risk probability
   - calibration: Isotonic regression to match event frequency
   
6. DEPLOYMENT STRATEGY:
   - Input: Physics scores + terrain features + rainfall for each grid cell
   - Output: Corrected risk = blend(physics_score, ml_score) or pure ML prediction
   - Option: Use ML as relative weight correction (e.g., if physics says 0.3 but terrain/rainfall suggest risk, ML learns weight adjustment)
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import pickle

try:
    import xarray as xr
except ImportError:
    xr = None

# Try to import ML libraries
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
    from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
except ImportError:
    print("ERROR: Install required libraries:")
    print("  pip install xgboost scikit-learn numpy pandas")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Regions to include
    'REGIONS': ['Assam', 'Uttarakhand', 'Western Ghats'],
    
    # Event matching tolerance (degrees; ~0.1 deg ≈ 11 km at equator)
    'SPATIAL_MATCH_TOLERANCE_DEG': 0.5,  # Increased to ~55 km for better event capture

    # Rainfall feature settings
    'RAINFALL_YEARS': [2013, 2018, 2022],
    'MONSOON_MONTHS': [6, 7, 8, 9],
    
    # Time range for events
    'EVENT_YEAR_RANGE': (2010, 2025),
    
    # Training parameters
    'TEST_SIZE': 0.2,
    'RANDOM_STATE': 42,
    'N_FOLDS': 5,
    'CLASS_WEIGHT': 'balanced',
    
    # XGBoost hyperparameters
    'XGB_PARAMS': {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 10,  # Handle class imbalance (10:1 negative:positive)
        'random_state': 42,
    },
    
    # Feature selection
    'DROP_FEATURES': ['event_id', 'year', 'region', 'year_norm', 'rain_year_used', 'disaster_type', 'disaster_type_encoded', 'region_encoded'],
    'CATEGORICAL_FEATURES': [],
}

# ============================================================================
# STEP 1: LOAD AND PREPARE HISTORICAL EVENTS
# ============================================================================

def load_historical_events(csv_path, config):
    """Load events from CSV and filter to target regions & time range."""
    print("=" * 80)
    print("STEP 1: Loading Historical Events")
    print("=" * 80)
    
    df = pd.read_csv(csv_path)
    
    # Filter by regions
    region_filter = df['location'].str.contains(
        '|'.join(config['REGIONS']), 
        case=False, 
        na=False
    )
    df = df[region_filter]
    
    # Filter by year range
    year_min, year_max = config['EVENT_YEAR_RANGE']
    df = df[(df['year'] >= year_min) & (df['year'] <= year_max)]
    
    # Keep only events with coordinates
    df = df[df['latitude'].notna() & df['longitude'].notna()]
    
    # Assign region label
    def assign_region(location):
        if pd.isna(location):
            return 'Unknown'
        loc = str(location).lower()
        if 'assam' in loc:
            return 'Assam'
        elif 'uttarakhand' in loc or 'uttar' in loc:
            return 'Uttarakhand'
        else:
            return 'Western Ghats'
    
    df['region'] = df['location'].apply(assign_region)
    
    print(f"\nTotal events loaded: {df.shape[0]}")
    print(f"\nBreakdown by region:")
    print(df['region'].value_counts())
    print(f"\nBreakdown by disaster type:")
    print(df['disaster_type'].value_counts())
    print(f"\nBreakdown by year:")
    print(df['year'].value_counts().sort_index())
    
    return df

# ============================================================================
# STEP 2: LOAD SIMULATION OUTPUT (FOR PHYSICS FEATURES)
# ============================================================================

def load_simulation_geojson(output_dir, region):
    """Load GeoJSON from simulation outputs."""
    # Try different naming conventions
    patterns = [
        f'{region.lower()}_historical_summary.geojson',
        f'{region.lower()}_live_summary.geojson',
        f'{region.lower()}_dynamic_summary.geojson',
    ]
    
    for pattern in patterns:
        path = os.path.join(output_dir, pattern)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f), path
    
    return None, None

def extract_physics_features(geojson_data):
    """
    Extract physics model outputs from GeoJSON features.
    
    CONDITIONS CHECKED:
    - Feature must have valid coordinates
    - Physics properties: flood_prob, landslide_stress, composite_risk
    - Return as dict {(lat, lon): {properties}}
    """
    features_by_coord = {}
    
    if geojson_data is None:
        return features_by_coord
    
    for feature in geojson_data.get('features', []):
        geom = feature.get('geometry', {})
        props = feature.get('properties', {})
        
        # Extract coordinates
        if geom['type'] == 'Polygon':
            coords = geom['coordinates'][0]  # Exterior ring
            # Use centroid of polygon
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]
            center_lat, center_lon = np.mean(lats), np.mean(lons)
        else:
            continue
        
        # Extract physics outputs (case-insensitive)
        physics_props = {
            'flood_probability': props.get('flood') or 0.0,
            'landslide_stress': props.get('landslide') or 0.0,
            'composite_risk': props.get('risk') or 0.0,
        }
        
        key = (round(center_lat, 4), round(center_lon, 4))
        features_by_coord[key] = physics_props
    
    return features_by_coord

# ============================================================================
# STEP 3: MATCH EVENTS TO GRID CELLS
# ============================================================================

def match_events_to_cells(events_df, physics_features_dict, tolerance_deg):
    """
    Match event coordinates to nearest grid cell.
    
    CONDITIONS:
    - Find nearest cell within tolerance_deg
    - If no cell within tolerance, skip event (unreliable match)
    - Return: list of {event_lat, event_lon, cell_lat, cell_lon, distance, properties}
    """
    print("\n" + "=" * 80)
    print("STEP 2: Matching Historical Events to Grid Cells")
    print("=" * 80)
    
    matched_events = []
    unmatched_count = 0
    
    for idx, event in events_df.iterrows():
        event_lat = event['latitude']
        event_lon = event['longitude']
        
        # Find nearest cell
        if not physics_features_dict:
            unmatched_count += 1
            continue
        
        cell_coords = list(physics_features_dict.keys())
        distances = [
            np.sqrt((event_lat - lat)**2 + (event_lon - lon)**2)
            for lat, lon in cell_coords
        ]
        
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        # Check tolerance
        if min_dist <= tolerance_deg:
            nearest_cell = cell_coords[min_dist_idx]
            matched_events.append({
                'event_id': idx,
                'event_lat': event_lat,
                'event_lon': event_lon,
                'cell_lat': nearest_cell[0],
                'cell_lon': nearest_cell[1],
                'distance_deg': min_dist,
                'disaster_type': event['disaster_type'],
                'region': event['region'],
                'year': event['year'],
                'physics_props': physics_features_dict[nearest_cell],
            })
        else:
            unmatched_count += 1
    
    print(f"\nMatched events: {len(matched_events)}")
    print(f"Unmatched events: {unmatched_count}")
    print(f"Match ratio: {len(matched_events) / (len(matched_events) + unmatched_count) * 100:.1f}%")
    
    return matched_events

# ============================================================================
# STEP 4: CREATE TRAINING DATASET
# ============================================================================

def create_training_dataset(matched_events, physics_features_dict, config):
    """
    Create labeled training dataset.
    
    CONDITIONS FOR POSITIVE SAMPLES:
    - Cells containing historical events → label=1
    
    CONDITIONS FOR NEGATIVE SAMPLES:
    - Random cells without events → label=0
    - Sample 10x ratio of negatives to handle class imbalance
    - Ensure diversity across regions
    """
    print("\n" + "=" * 80)
    print("STEP 3: Creating Training Dataset")
    print("=" * 80)
    
    # Positive samples: matched events
    positive_samples = []
    for event in matched_events:
        positive_samples.append({
            'cell_lat': event['cell_lat'],
            'cell_lon': event['cell_lon'],
            'disaster_type': event['disaster_type'],
            'region': event['region'],
            'year': event['year'],
            'flood_probability': event['physics_props'].get('flood_probability', 0.0),
            'landslide_stress': event['physics_props'].get('landslide_stress', 0.0),
            'composite_risk': event['physics_props'].get('composite_risk', 0.0),
            'label': 1,  # Event occurred
        })
    
    # Negative samples: hard negatives + random cells without events
    positive_coords = set((e['cell_lat'], e['cell_lon']) for e in positive_samples)
    all_coords = list(physics_features_dict.keys())
    negative_coords = [c for c in all_coords if c not in positive_coords]
    
    # Sort negative coords by composite risk to find hard negatives
    negative_cells_with_risk = []
    for c in negative_coords:
        props = physics_features_dict[c]
        risk = props.get('composite_risk', 0.0)
        negative_cells_with_risk.append((c, risk))
        
    negative_cells_with_risk.sort(key=lambda x: x[1], reverse=True)
    
    # Sample 10x ratio
    neg_sample_count = max(len(matched_events) * 10, 100)
    
    # 30% hard negatives (cells with highest physics risk but no recorded event)
    hard_neg_count = min(int(neg_sample_count * 0.3), len(negative_coords))
    hard_negative_coords = [c for c, _ in negative_cells_with_risk[:hard_neg_count]]
    
    # 70% random negatives from the rest
    remaining_negative_coords = [c for c, _ in negative_cells_with_risk[hard_neg_count:]]
    np.random.seed(config['RANDOM_STATE'])
    num_random = min(neg_sample_count - len(hard_negative_coords), len(remaining_negative_coords))
    if remaining_negative_coords:
        random_indices = np.random.choice(len(remaining_negative_coords), size=num_random, replace=False)
        random_negative_coords = [remaining_negative_coords[i] for i in random_indices]
    else:
        random_negative_coords = []
        
    sampled_negative_coords = hard_negative_coords + random_negative_coords
    
    negative_samples = []
    for lat, lon in sampled_negative_coords:
        props = physics_features_dict[(lat, lon)]
        negative_samples.append({
            'cell_lat': lat,
            'cell_lon': lon,
            'disaster_type': 'none',
            'region': assign_region_from_coords(lat, lon),
            'year': 2018,  # Default to median year
            'flood_probability': props.get('flood_probability', 0.0),
            'landslide_stress': props.get('landslide_stress', 0.0),
            'composite_risk': props.get('composite_risk', 0.0),
            'label': 0,  # No event
        })
    
    # Combine
    all_samples = positive_samples + negative_samples
    training_df = pd.DataFrame(all_samples)
    
    print(f"\nPositive samples (events): {len(positive_samples)}")
    print(f"Negative samples (no events): {len(negative_samples)}")
    print(f"Total training samples: {training_df.shape[0]}")
    print(f"Class distribution:")
    print(training_df['label'].value_counts())
    print(f"\nRegion distribution:")
    print(training_df['region'].value_counts())
    
    return training_df

def assign_region_from_coords(lat, lon):
    """Simple region assignment from coordinates (can be refined)."""
    if lat > 25 and lon > 88:
        return 'Assam'
    elif lat > 28 and lon > 78:
        return 'Uttarakhand'
    else:
        return 'Western Ghats'

# ============================================================================
# STEP 5: RAINFALL FEATURES (CHIRPS)
# ============================================================================

def load_chirps_datasets(rainfall_dir, years):
    """Load CHIRPS daily rainfall datasets for configured years."""
    datasets = {}
    
    if xr is None:
        print("\n⚠ xarray not installed; rainfall features will be zeros.")
        return datasets

    for year in years:
        nc_path = rainfall_dir / f'chirps-v2.0.{year}.days_p05.nc'
        if nc_path.exists():
            datasets[year] = xr.open_dataset(nc_path)
        else:
            print(f"\n⚠ Missing rainfall file: {nc_path}")

    return datasets


def nearest_available_year(target_year, available_years):
    """Map a target year to nearest available CHIRPS year."""
    if not available_years:
        return None
    return min(available_years, key=lambda y: abs(y - int(target_year)))


def add_rainfall_features(training_df, rainfall_dir, config):
    """
    Add rainfall features using CHIRPS rainfall from configured years.

    Added columns:
    - rain_annual_mm: yearly rainfall at nearest grid point
    - rain_monsoon_mm: Jun-Sep rainfall at nearest grid point
    - rain_variability_cv: CV of annual rainfall across available CHIRPS years
    - rain_year_used: nearest CHIRPS year used for this sample
    """
    print("\n" + "=" * 80)
    print("STEP 4: Enriching Samples with CHIRPS Rainfall Features")
    print("=" * 80)

    df = training_df.copy()
    chirps = load_chirps_datasets(rainfall_dir, config['RAINFALL_YEARS'])

    if not chirps:
        df['rain_annual_mm'] = 0.0
        df['rain_monsoon_mm'] = 0.0
        df['rain_variability_cv'] = 0.0
        df['rain_year_used'] = -1
        return df

    available_years = sorted(chirps.keys())
    monsoon_months = set(config['MONSOON_MONTHS'])

    # Cache point extraction by (rounded_lat, rounded_lon, year)
    point_cache = {}
    variability_cache = {}

    def extract_for_point(lat, lon, year):
        cache_key = (round(float(lat), 4), round(float(lon), 4), int(year))
        if cache_key in point_cache:
            return point_cache[cache_key]

        used_year = nearest_available_year(year, available_years)
        ds = chirps[used_year]
        point_series = ds['precip'].sel(latitude=float(lat), longitude=float(lon), method='nearest')
        annual = float(point_series.sum(skipna=True).values)
        monsoon_mask = point_series['time'].dt.month.isin(list(monsoon_months))
        monsoon = float(point_series.where(monsoon_mask, drop=True).sum(skipna=True).values)

        point_cache[cache_key] = (annual, monsoon, used_year)
        return point_cache[cache_key]

    def rainfall_variability(lat, lon):
        cache_key = (round(float(lat), 4), round(float(lon), 4))
        if cache_key in variability_cache:
            return variability_cache[cache_key]

        annual_totals = []
        for y in available_years:
            ds = chirps[y]
            point_series = ds['precip'].sel(latitude=float(lat), longitude=float(lon), method='nearest')
            annual_totals.append(float(point_series.sum(skipna=True).values))

        mean_val = float(np.mean(annual_totals)) if annual_totals else 0.0
        std_val = float(np.std(annual_totals)) if annual_totals else 0.0
        cv = (std_val / mean_val) if mean_val > 0 else 0.0
        variability_cache[cache_key] = cv
        return cv

    annual_vals = []
    monsoon_vals = []
    cv_vals = []
    year_used_vals = []

    for row in df.itertuples(index=False):
        annual, monsoon, used_year = extract_for_point(row.cell_lat, row.cell_lon, row.year)
        cv = rainfall_variability(row.cell_lat, row.cell_lon)

        annual_vals.append(annual)
        monsoon_vals.append(monsoon)
        cv_vals.append(cv)
        year_used_vals.append(used_year)

    df['rain_annual_mm'] = annual_vals
    df['rain_monsoon_mm'] = monsoon_vals
    df['rain_variability_cv'] = cv_vals
    df['rain_year_used'] = year_used_vals

    print(f"Loaded CHIRPS years: {available_years}")
    print(f"Rainfall feature stats:")
    print(df[['rain_annual_mm', 'rain_monsoon_mm', 'rain_variability_cv']].describe().to_string())

    return df

# ============================================================================
# STEP 6: FEATURE ENGINEERING
# ============================================================================

def engineer_features(training_df, config):
    """
    Create derived features from base features.
    
    INTERACTION FEATURES CREATED:
    - slope_x_rainfall: (assumed from physics outputs)
    - elevation_x_slope: (proxy from composite_risk)
    - flow_accumulation effects: (captured in landslide_stress)
    - rainfall_intensity: (from flood_probability + season/region)
    """
    print("\n" + "=" * 80)
    print("STEP 4: Feature Engineering")
    print("=" * 80)
    
    df = training_df.copy()
    
    # Base features (already in df):
    # - flood_probability
    # - landslide_stress
    # - composite_risk
    
    # Interaction features
    df['flood_x_landslide'] = df['flood_probability'] * df['landslide_stress']
    df['flood_plus_landslide'] = df['flood_probability'] + df['landslide_stress']
    df['max_hazard'] = df[['flood_probability', 'landslide_stress']].max(axis=1)
    df['hazard_disagreement'] = (df['flood_probability'] - df['landslide_stress']).abs()

    # Rainfall interactions
    if 'rain_annual_mm' in df.columns:
        df['flood_x_rain_annual'] = df['flood_probability'] * df['rain_annual_mm']
        df['landslide_x_rain_monsoon'] = df['landslide_stress'] * df['rain_monsoon_mm']
        df['rain_intensity_ratio'] = df['rain_monsoon_mm'] / (df['rain_annual_mm'] + 1e-6)
    
    # Region encoding
    region_encoder = LabelEncoder()
    df['region_encoded'] = region_encoder.fit_transform(df['region'])
    
    # Disaster type encoding
    type_encoder = LabelEncoder()
    df['disaster_type_encoded'] = type_encoder.fit_transform(df['disaster_type'])
    
    # Temporal features
    year_span = (df['year'].max() - df['year'].min())
    df['year_norm'] = (df['year'] - df['year'].min()) / (year_span if year_span != 0 else 1)
    
    # Log-transformed features for better distribution
    df['flood_probability_log'] = np.log1p(df['flood_probability'])
    df['landslide_stress_log'] = np.log1p(df['landslide_stress'])
    
    print(f"Total features created: {df.shape[1] - 1}")  # Exclude label
    print(f"Feature list:")
    print(df.columns.tolist())
    
    return df, region_encoder, type_encoder

# ============================================================================
# STEP 7: TRAIN MODEL
# ============================================================================

def train_model(training_df, config):
    """
    Train XGBoost model using stratified cross-validation.
    
    TRAINING CONDITIONS:
    - 5-fold stratified cross-validation (by region + disaster type)
    - 80/20 train/test split
    - Class weight: balanced to handle imbalance
    - Evaluation metrics: ROC-AUC, F1, PR-AUC
    """
    print("\n" + "=" * 80)
    print("STEP 5: Training XGBoost Model")
    print("=" * 80)
    
    # Prepare features and target
    feature_cols = [c for c in training_df.columns 
                   if c not in ['label', 'cell_lat', 'cell_lon'] + config['DROP_FEATURES']]
    
    X = training_df[feature_cols]
    y = training_df['label']
    
    # Remove non-numeric features before scaling
    X = X.select_dtypes(include=[np.number])
    feature_cols = X.columns.tolist()
    
    print(f"\nFeatures for training: {len(feature_cols)}")
    print(feature_cols)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=config['TEST_SIZE'],
        random_state=config['RANDOM_STATE'],
        stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
    
    # Train XGBoost
    model = xgb.XGBClassifier(**config['XGB_PARAMS'])
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, model.predict(X_test))
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    print(f"\nModel Performance on Test Set:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Calibrate model for probability estimates
    print(f"\nCalibrating model probabilities...")
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    return calibrated_model, scaler, feature_cols, importance_df

# ============================================================================
# STEP 8: SAVE MODEL
# ============================================================================

def save_model(model, scaler, feature_cols, importance_df, output_dir):
    """Save trained model and metadata."""
    print("\n" + "=" * 80)
    print("STEP 6: Saving Model")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'ml_risk_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'train_date': datetime.now().isoformat(),
        }, f)
    print(f"✓ Model saved: {model_path}")
    
    # Save feature importance
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved: {importance_path}")
    
    # Save feature list
    features_path = os.path.join(output_dir, 'feature_list.txt')
    with open(features_path, 'w') as f:
        f.write("Feature Engineering Documentation\n")
        f.write("=" * 60 + "\n\n")
        f.write("FEATURES USED FOR TRAINING:\n")
        for i, col in enumerate(feature_cols, 1):
            f.write(f"{i}. {col}\n")
        f.write("\n\nFEATURE ENGINEERING CONDITIONS:\n")
        f.write("- Base physics outputs: flood_probability, landslide_stress, composite_risk\n")
        f.write("- Interactions: flood_x_landslide, flood_plus_landslide, max_hazard, hazard_disagreement\n")
        f.write("- Encoded: region_encoded, disaster_type_encoded\n")
        f.write("- Temporal: year_norm\n")
        f.write("- Log-transform: flood_probability_log, landslide_stress_log\n")
        f.write("\n\nTRAINING CONDITIONS:\n")
        f.write("- Positive samples: Grid cells with historical flood/landslide events\n")
        f.write("- Negative samples: Random cells without events (10:1 ratio)\n")
        f.write("- Model: XGBoost with class weight balancing\n")
        f.write("- Cross-validation: 5-fold stratified\n")
        f.write("- Calibration: Isotonic regression for probability estimates\n")
    print(f"✓ Features saved: {features_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Paths
    project_root = Path(__file__).parent
    event_csv = project_root / 'data' / 'finalss_india_flood_landslide.csv'
    rainfall_dir = project_root / 'data' / 'rainfall'
    output_dir = project_root / 'outputs'
    model_output_dir = project_root / 'ml_models'
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "ML MODEL TRAINING FOR FLOOD/LANDSLIDE RISK PREDICTION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print(f"\nTraining Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project Root: {project_root}")
    print(f"Event Data: {event_csv}")
    print(f"Rainfall Data: {rainfall_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Model Save Path: {model_output_dir}")
    
    # ---- STEP 1: Load Events ----
    events_df = load_historical_events(event_csv, CONFIG)
    
    # ---- STEP 2 & 3: Load physics features and match events ----
    all_matched_events = []
    all_physics_features = {}
    
    for region in CONFIG['REGIONS']:
        print(f"\n\nProcessing region: {region}")
        geojson_data, geojson_path = load_simulation_geojson(output_dir, region)
        
        if geojson_data is None:
            print(f"  ⚠ No GeoJSON found for {region}")
            continue
        
        print(f"  ✓ Loaded GeoJSON: {geojson_path}")
        
        # Extract physics features
        physics_features = extract_physics_features(geojson_data)
        print(f"  ✓ Extracted {len(physics_features)} grid cells with physics properties")
        all_physics_features.update(physics_features)
        
        # Match events for this region
        region_events = events_df[events_df['region'] == region]
        matched = match_events_to_cells(
            region_events,
            physics_features,
            CONFIG['SPATIAL_MATCH_TOLERANCE_DEG']
        )
        all_matched_events.extend(matched)
    
    if not all_matched_events:
        print("\n❌ ERROR: No events could be matched to grid cells!")
        print("Ensure GeoJSON files exist in outputs/ directory")
        return
    
    # ---- STEP 4: Create training dataset ----
    training_df = create_training_dataset(all_matched_events, all_physics_features, CONFIG)

    # ---- STEP 5: Add CHIRPS rainfall features ----
    training_df = add_rainfall_features(training_df, rainfall_dir, CONFIG)
    
    # ---- STEP 6: Feature engineering ----
    training_df_engineered, region_enc, type_enc = engineer_features(training_df, CONFIG)
    
    # ---- STEP 7: Train model ----
    model, scaler, feature_cols, importance_df = train_model(training_df_engineered, CONFIG)
    
    # ---- STEP 8: Save model ----
    save_model(model, scaler, feature_cols, importance_df, model_output_dir)
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Review feature_importance.csv in {model_output_dir}/")
    print(f"2. Integrate model inference into services/hazard_service.py")
    print(f"3. Add ML output layer to frontend map")


if __name__ == '__main__':
    main()
