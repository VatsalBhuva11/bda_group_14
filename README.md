# Emission Hotspot Dashboard

A Streamlit dashboard to explore the merged pollution-vehicle dataset, visualize KPIs and insights, and interact with a prediction playground using your trained ML model.

## Features
- Overview KPIs with units and regional hotspot rates
- Sidebar filters for regions, make, class, and numeric ranges
- Interactive Plotly charts with drill-down grouping and metric selection
- Prediction Playground using saved encoders, scaler, and model
- Data Explorer with CSV download of filtered data

## Project Structure
- `app.py` - Streamlit dashboard application
- `comprehensive_pollution_vehicle_dataset.csv` - merged dataset
- `models/` - directory containing saved model artifacts
  - `emission_hotspot_model.pkl`
  - `feature_scaler.pkl`
  - `vehicle_class_encoder.pkl`
  - `vehicle_make_encoder.pkl`
  - `economic_region_encoder.pkl`
- `requirements.txt` - Python dependencies

## Setup
1. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure files are present:
- `comprehensive_pollution_vehicle_dataset.csv` in repository root
- Model artifacts in `models/` as listed above (optional but recommended)

## Run
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Notes
- If the `Hotspot` column is missing, the app attempts to derive it from a score column at the 70th percentile.
- The Prediction Playground aligns features to the scaler's `feature_names_in_` when available.
- For best results, provide the exact encoders/scaler used during training. 