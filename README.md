# Listeria Soil Risk Estimator (Streamlit Prototype)

This app is a **prototype decision-support tool** built from the information available in the uploaded RiskForgers report and README.

## What it does
- Lets a user choose a **state**
- Lets a user enter key environmental and land-use factors
- Returns:
  - estimated **probability of Listeria positivity**
  - estimated **mean concentration (CFU/g)**
  - estimated **95th percentile concentration (CFU/g)**

## What it is based on
1. The reported **state-level prevalence and concentration table**
2. The strongest reported risk drivers:
   - moisture
   - cropland
   - pasture
   - shrubland
   - longitude
   - precipitation / humidity
   - temperature extremes
   - elevation
   - proximity to surface water

## Important limitation
This app does **not yet** run the original trained model directly. It uses a transparent scoring engine informed by the uploaded project summary.

## Best next step
Export the fitted model artifacts from the competition pipeline and wire them into `app.py`.

Suggested artifacts:
- `final_model.pkl`
- `preprocessor.pkl`
- `feature_order.json`
- optional calibration object for concentration conversion

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
