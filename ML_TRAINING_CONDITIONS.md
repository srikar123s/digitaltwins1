# ML Model Training - Complete Conditions & Feature Engineering

## Overview
This document specifies **all conditions, features, and logic** used in the ML model training pipeline for flood/landslide risk prediction across **all regions** (Assam, Uttarakhand, Western Ghats).

---

## 1. DATA COLLECTION & FILTERING CONDITIONS

### 1.1 Historical Events
**Source:** `data/finalss_india_flood_landslide.csv`

**Filtering Conditions:**
- ✓ Geographic: Events in Assam, Uttarakhand, or Western Ghats (location name match)
- ✓ Temporal: Years 2010–2025 (overlap with available rainfall & terrain data)
- ✓ Coordinates: Must have non-null latitude & longitude
- ✓ Disaster Type: Both floods and landslides included

**Current Dataset:**
- Total events available: 2,840
- Events in target regions: 109 (Assam) + 30 (Uttarakhand) + 0 (Western Ghats) = 139
- Events in 2010–2025: 61 total
- With valid coordinates: **29 events**
  - Assam: 18 events
  - Uttarakhand: 11 events
  - Western Ghats: 0 events (fallback: include if available in other data)

### 1.2 Simulation/GeoJSON Data
**Source:** `outputs/{region}_{mode}_summary.geojson` (e.g., `assam_historical_summary.geojson`)

**Expected Content per cell:**
- Geometry: Polygon (grid cell boundary)
- Properties: Flood probability, landslide stress, composite risk score

**Loading Conditions:**
- ✓ File must exist for region
- ✓ Each feature must have coordinates (polygon centroid computed)
- ✓ Physics properties extracted or defaulted to 0.0

---

## 2. EVENT-TO-GRID MATCHING CONDITIONS

### 2.1 Spatial Matching
**Match Tolerance:** 0.1 degrees (~11 km at equator)

**Logic:**
```
For each event (lat, lon):
  1. Find all grid cells with their centroids
  2. Compute distance to each cell centroid
  3. If minimum distance ≤ 0.1°:
     → Match event to nearest cell as POSITIVE sample
     → Label = 1 (hazard event occurred)
  4. Else:
     → Skip event (unreliable match)
```

**Example:**
- Event at (26.2350°N, 92.2500°E) matched to cell (26.2348°N, 92.2499°E) ✓ (distance < 0.01°)
- Event at (25.0°N, 90.0°E) → no cell within 0.1° → skipped ✗

### 2.2 Outcome
- Expected matched events: ~25–29 (from available data)
- Matched events → Positive training samples (label=1)

---

## 3. TRAINING DATASET CONSTRUCTION

### 3.1 Positive Samples (label=1)
**Source:** Matched historical events

**Conditions:**
- One sample per matched event
- Properties:
  - `cell_lat, cell_lon`: Grid cell location
  - `disaster_type`: 'flood' or 'landslide'
  - `region`: Assam, Uttarakhand, or Western Ghats
  - `year`: Year of event
  - Physics outputs: flood_probability, landslide_stress, composite_risk
  - `label`: 1 (event occurred)

**Expected Count:** ~25–29 samples

### 3.2 Negative Samples (label=0)
**Source:** Random grid cells WITHOUT events

**Conditions:**
- Random selection from all grid cells that did NOT have events
- Class balance ratio: 10 negative per 1 positive (10:1 ratio)
- Ensures model learns background distribution
- Sampled from all regions equally

**Expected Count:** ~250–290 samples

**Total Training Data:** ~280–320 samples

### 3.3 Class Imbalance Handling
**Method:** XGBoost `scale_pos_weight=10`
- Automatically weights positive class 10x heavier during training
- Prevents model bias toward majority class

---

## 4. FEATURE ENGINEERING

### 4.1 Base Features (Physics Model Outputs)
**Direct from GeoJSON properties:**

| Feature | Type | Range | Meaning |
|---------|------|-------|---------|
| `flood_probability` | Float | [0, 1] | Physics model: flood risk score |
| `landslide_stress` | Float | [0, 1] | Physics model: landslide susceptibility |
| `composite_risk` | Float | [0, 1] | Physics model: combined hazard |

### 4.2 Interaction Features
**Derived combinations:**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `flood_x_landslide` | flood × landslide | Cells with both hazards amplify risk |
| `flood_plus_landslide` | flood + landslide | Simple additive hazard |
| `max_hazard` | max(flood, landslide) | Worst-case scenario |
| `hazard_disagreement` | \|flood - landslide\| | Model consensus metric |

**Interpretation:**
- High `flood_x_landslide` → Both models agree on risk
- High `hazard_disagreement` → Models conflict (needs ML correction)

### 4.3 Categorical Features (Encoded)

| Feature | Type | Values | Purpose |
|---------|------|--------|---------|
| `region_encoded` | Ordinal | 0—Assam, 1—Uttarakhand, 2—Western Ghats | Regional variation |
| `disaster_type_encoded` | Ordinal | 0—none, 1—flood, 2—landslide | Hazard specialization |

### 4.4 Temporal Features

| Feature | Type | Range | Purpose |
|---------|------|-------|---------|
| `year_norm` | Float | [0, 1] | Normalized year (2010 → 0, 2025 → 1) |

**Rationale:** Climate patterns & infrastructure change over time

### 4.5 Log-Transformed Features
**For better ML learning:**

| Feature | Formula |
|---------|---------|
| `flood_probability_log` | log(1 + flood_probability) |
| `landslide_stress_log` | log(1 + landslide_stress) |

**Rationale:** Makes relationships more linear for tree-based models

### 4.6 Summary: Final Feature Count
**Total features: ~15–18**
- Physics base: 3
- Interactions: 4
- Encoded: 2
- Temporal: 1
- Log-transformed: 2
- Derived/neighborhood: ~3 (if available)

---

## 5. MODEL SELECTION & ARCHITECTURE

### 5.1 Algorithm: XGBoost
**Why XGBoost?**
- ✓ Handles non-linear interactions (flood × rainfall × slope)
- ✓ Built-in feature importance ranking
- ✓ Robust to missing values & outliers
- ✓ Fast training on small-to-medium data
- ✓ Interpretable feature contributions

### 5.2 Hyperparameters
```python
{
    'objective': 'binary:logistic',     # Probability prediction
    'max_depth': 6,                      # Tree depth (avoid overfitting)
    'learning_rate': 0.1,                # Gradient update rate
    'n_estimators': 200,                 # Number of trees
    'subsample': 0.8,                    # Row sampling (regularization)
    'colsample_bytree': 0.8,             # Feature sampling
    'scale_pos_weight': 10,              # Handle class imbalance
    'random_state': 42,                  # Reproducibility
}
```

### 5.3 Output
**Prediction:** `ml_risk_score` ∈ [0, 1]
- 0 = No hazard risk
- 1 = Maximum hazard risk
- Calibrated via Isotonic Regression for probability estimates

---

## 6. TRAINING & VALIDATION CONDITIONS

### 6.1 Data Split
**Train/Test:** 80% / 20%
- Stratified by label (preserve class ratio)
- Prevents temporal leakage (no future events in train set)

**Cross-Validation:** 5-fold stratified
- Each fold preserves label distribution
- Per-fold metrics to check model stability

### 6.2 Evaluation Metrics
**Test Set Performance:**

| Metric | Ideal | Interpretation |
|--------|-------|-----------------|
| **ROC-AUC** | 0.8–0.95 | Overall discrimination ability |
| **PR-AUC** | 0.7–0.9 | Precision-Recall for imbalanced data |
| **F1-Score** | 0.6–0.8 | Balance of precision & recall |

**Target:** ROC-AUC ≥ 0.80 (good discrimination between events & non-events)

### 6.3 Calibration
**Method:** Isotonic Regression
- Ensures predicted probabilities match actual event frequency
- Example: If model predicts 0.6 probability, ~60% of those cells should have events

---

## 7. TRAINING CONDITIONS CHECKLIST

- [ ] Event CSV exists at `data/finalss_india_flood_landslide.csv`
- [ ] GeoJSON files exist for each region in `outputs/`
  - [ ] Assam (e.g., `assam_historical_summary.geojson`)
  - [ ] Uttarakhand (e.g., `uttarakhand_historical_summary.geojson`)
  - [ ] Western Ghats (fallback pattern matching)
- [ ] Physics properties in GeoJSON: `flood_prob`, `landslide_stress`, `composite_risk`
- [ ] Event coordinates are non-null (latitude, longitude)
- [ ] Spatial match tolerance: 0.1 degrees (~11 km)
- [ ] Class imbalance handled: 10:1 negative:positive ratio
- [ ] Feature scaling: StandardScaler applied to all features
- [ ] XGBoost hyperparameters: max_depth=6, learning_rate=0.1, n_estimators=200
- [ ] Validation: 5-fold cross-validation with stratification
- [ ] Calibration: Isotonic Regression for probability estimates

---

## 8. EXPECTED RESULTS

### 8.1 Training Output Files
```
ml_models/
├── ml_risk_model.pkl              # Trained model + scaler + feature list
├── feature_importance.csv         # Feature contribution rankings
└── feature_list.txt               # Documentation of all features
```

### 8.2 Model Artifacts
- **Model object:** XGBoost classifier (200 trees, depth 6)
- **Scaler:** StandardScaler fitted on training data
- **Feature columns:** List of all ~15–18 features
- **Metadata:** Training date, region coverage, sample counts

### 8.3 Expected Feature Importance (Top 5)
1. `composite_risk` (~25%) – Physics model already captures risk well
2. `flood_probability` (~20%) – Direct hazard indicator
3. `hazard_disagreement` (~15%) – Model conflict signals need for ML
4. `landslide_stress` (~12%) – Secondary hazard
5. `region_encoded` (~10%) – Regional variation in risk patterns

---

## 9. LIMITATIONS & FUTURE IMPROVEMENTS

### Current Limitations
- ✗ Small dataset (25–29 positive samples) → Risk of overfitting
- ✗ Western Ghats has 0 labeled events in period → May need all-region training
- ✗ Missing detailed rainfall data per cell → Could add CHIRPS integration
- ✗ No terrain elevation/slope in features (if not in GeoJSON)

### Future Improvements
- Add terrain features: elevation, slope, aspect from DEM
- Integrate CHIRPS rainfall data (2013, 2018, 2022)
- Expand event dataset: Include pre-2010 or post-2025 with care
- Online learning: Gradually retrain model as new events occur
- Uncertainty quantification: Add prediction confidence intervals
- Per-region sub-models: Train separate models for Assam vs. Uttarakhand

---

## 10. RUNNING THE TRAINING SCRIPT

```bash
cd c:\Users\srikar\OneDrive\Desktop\all projects\digitaltwins

# Activate Python environment
.\venv\Scripts\Activate.ps1

# Install required libraries (if needed)
pip install xgboost scikit-learn numpy pandas

# Run training
python train_ml_model.py
```

**Expected Output:**
- Step 1: Load 29 historical events (Assam + Uttarakhand)
- Step 2: Match events to grid cells (tolerance 0.1°)
- Step 3: Create training set (25–29 positive, 250–290 negative samples)
- Step 4: Engineer features (15–18 features)
- Step 5: Train XGBoost, show ROC-AUC in test set
- Step 6: Save model to `ml_models/`

**Training time:** ~30–60 seconds on standard laptop

---

## 11. DEPLOYMENT INTEGRATION

Once model is trained, integrate into backend:

**[Backend] services/hazard_service.py**
```python
def predict_ml_risk(grid_cells, physics_features):
    """Load trained model and predict ML risk for each cell."""
    model = load_model('ml_models/ml_risk_model.pkl')
    ml_scores = model.predict_proba(features)[:, 1]
    return ml_scores

def blend_predictions(physics_risk, ml_risk, blend_factor=0.6):
    """Combine physics + ML for hybrid prediction."""
    return blend_factor * ml_risk + (1 - blend_factor) * physics_risk
```

**[Frontend] frontend/src/App.jsx**
```javascript
// Add toggle for Physics vs. Hybrid ML output
const [outputMode, setOutputMode] = useState('physics'); // 'physics' | 'ml' | 'hybrid'

// Filter & render based on mode
const displayRisk = outputMode === 'physics' ? feature.physics_risk :
                    outputMode === 'ml' ? feature.ml_risk :
                    feature.hybrid_risk;
```

---

**Training Completed:** April 5, 2026
**Author:** ML Pipeline
**Version:** 1.0
