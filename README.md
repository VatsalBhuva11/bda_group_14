# Emission Hotspot Dashboard

A Streamlit dashboard to explore the merged pollution–vehicle dataset, visualize KPIs, and interact with an ML prediction playground.

## Features
- Overview KPIs with units and hotspot rate
- Sidebar filters for year, region, economic region, make, class, and numeric ranges
- Interactive Plotly charts (overview, time series, geography, vehicles)
- Prediction Playground using saved encoders, scaler, and model
- Data Explorer with CSV download of filtered data

## Project Structure
- `app.py` – Streamlit application
- `comprehensive_pollution_vehicle_dataset.csv` – merged dataset
- `models/` – saved artifacts (saved/loaded via joblib)
  - `emission_hotspot_model.pkl`
  - `feature_scaler.pkl`
  - `vehicle_class_encoder.pkl`
  - `vehicle_make_encoder.pkl`
  - `economic_region_encoder.pkl`
- `regenerate_models.py` – script to train and export artifacts
- `requirements.txt` – dependencies

## Setup
1) Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:
```bash
pip install -r requirements.txt
```

3) Ensure files are present:
- `comprehensive_pollution_vehicle_dataset.csv` in the repository root
- Artifacts in `models/` (you can also regenerate them; see below)

## Run
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

## Regenerating Models
If the models are missing or outdated, regenerate them with:
```bash
python regenerate_models.py
```
This script trains a classifier, creates the scaler and label encoders, and saves all artifacts into `models/` using `joblib.dump`.

## Troubleshooting
- Error: `invalid load key, '\x0a'` or model not found in the app
  - Artifacts are loaded with joblib. If you previously saved with another method or files got corrupted, run:
    ```bash
    python regenerate_models.py
    streamlit cache clear
    ```
    Then start the app again: `streamlit run app.py`.
- Predictions tab empty or stale
  - Clear Streamlit cache as above to refresh cached resources.

## Notes
- The Prediction Playground aligns inputs to the scaler’s expected features and includes `Year` so scaling matches training.
- Higher PM2.5, CO₂, fuel consumption, engine size, and market share generally increase hotspot risk.
- For non-extreme probabilities, try values near the decision boundary (e.g., PM2.5≈40–50, CO₂≈250–300, Fuel≈10–12, Engine≈2.3–2.8, Market≈0.18–0.25, Year≈2015–2018).

## Example Inputs
- Normal:
  - PM2.5: 15, CO₂: 160, Fuel: 6.5, Engine: 1.6, Market: 0.06, Year: 2013
- Hotspot (borderline):
  - PM2.5: 45, CO₂: 280, Fuel: 11.0, Engine: 2.6, Market: 0.22, Year: 2017
- Hotspot (clear):
  - PM2.5: 75, CO₂: 420, Fuel: 18.0, Engine: 5.0, Market: 0.55, Year: 2019