from __future__ import annotations

from pathlib import Path
import math
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / 'data' / 'state_estimates.csv'

st.set_page_config(
    page_title='Listeria Soil Risk Estimator',
    page_icon='🧪',
    layout='wide',
)


@st.cache_data
def load_state_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    num_cols = ['n', 'prevalence', 'p_upper', 'cs_li_mean_cfu_g', 'sim_q95_cfu_g']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_values('state').reset_index(drop=True)


def logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return math.log(p / (1 - p))


def logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def estimate_risk(
    baseline_prevalence: float,
    p_upper: float,
    baseline_mean: float,
    baseline_q95: float,
    inputs: Dict[str, float | str | bool],
) -> Dict[str, float | str | pd.DataFrame]:
    """
    Transparent prototype score built from directional findings in the uploaded
    report/README. It is intentionally not presented as the exact trained model.
    """

    # For states with observed prevalence = 0, use a conservative non-zero modeling anchor.
    # This lets the scenario engine respond to user inputs while preserving the displayed
    # baseline table values exactly as reported.
    anchor_p = baseline_prevalence if baseline_prevalence > 0 else max(0.01, p_upper / 2)

    moisture_map = {'Low': -0.35, 'Medium': 0.0, 'High': 0.35}
    near_water = 0.30 if inputs['near_water'] else 0.0

    contributions = {
        'State baseline': 0.0,
        'Moisture': moisture_map[inputs['moisture_level']],
        'Near water body': near_water,
        'Cropland %': clamp(0.35 * ((inputs['cropland_pct'] - 10) / 40), -0.30, 0.60),
        'Pasture %': clamp(0.25 * ((inputs['pasture_pct'] - 10) / 40), -0.20, 0.45),
        'Shrubland %': clamp(-0.40 * ((inputs['shrubland_pct'] - 10) / 40), -0.60, 0.35),
        'Max RH': clamp(0.20 * ((inputs['rh_max'] - 70) / 20), -0.20, 0.35),
        'Rainfall (mm)': clamp(0.20 * ((inputs['ppt_sum_mm'] - 5) / 20), -0.15, 0.35),
        'Mean temp (°C)': clamp(-0.18 * ((inputs['temp_mean_c'] - 18) / 12), -0.30, 0.25),
        'Max temp (°C)': clamp(0.18 * ((inputs['temp_max_c'] - 30) / 10), -0.20, 0.35),
        'Wind mean (m/s)': clamp(0.08 * ((inputs['wind_mean_ms'] - 3) / 4), -0.08, 0.15),
        'Elevation (m)': clamp(-0.20 * ((inputs['elevation_m'] - 300) / 1000), -0.35, 0.10),
        'Longitude': clamp(0.25 * ((inputs['longitude'] + 95) / 15), -0.30, 0.30),
    }

    delta = sum(v for k, v in contributions.items() if k != 'State baseline')
    adjusted_probability = logistic(logit(anchor_p) + delta)

    # Scale concentration by probability shift, but keep multipliers bounded.
    multiplier = clamp(adjusted_probability / max(anchor_p, 0.05), 0.50, 3.00)
    adj_mean = baseline_mean * multiplier
    adj_q95 = baseline_q95 * multiplier

    if adjusted_probability >= 0.70:
        band = 'High'
    elif adjusted_probability >= 0.40:
        band = 'Moderate'
    else:
        band = 'Lower'

    contrib_df = pd.DataFrame(
        {'Factor': list(contributions.keys())[1:], 'Log-odds shift': list(contributions.values())[1:]}
    ).sort_values('Log-odds shift')

    return {
        'anchor_p': anchor_p,
        'adjusted_probability': adjusted_probability,
        'adjusted_mean': adj_mean,
        'adjusted_q95': adj_q95,
        'multiplier': multiplier,
        'risk_band': band,
        'contributions': contrib_df,
    }


def download_payload(state_row: pd.Series, scenario: Dict[str, float | str], inputs: Dict[str, float | str | bool]) -> bytes:
    payload = {
        'state': state_row['state'],
        'reported_baseline': {
            'n': int(state_row['n']),
            'prevalence': float(state_row['prevalence']),
            'p_upper': float(state_row['p_upper']),
            'ci_95': state_row['ci_95'],
            'cs_li_mean_cfu_g': float(state_row['cs_li_mean_cfu_g']),
            'sim_q95_cfu_g': float(state_row['sim_q95_cfu_g']),
        },
        'user_inputs': inputs,
        'prototype_scenario_output': {
            'adjusted_probability': float(scenario['adjusted_probability']),
            'adjusted_mean_cfu_g': float(scenario['adjusted_mean']),
            'adjusted_q95_cfu_g': float(scenario['adjusted_q95']),
            'multiplier_vs_state_baseline': float(scenario['multiplier']),
            'risk_band': scenario['risk_band'],
        },
    }
    return json.dumps(payload, indent=2).encode('utf-8')


def main() -> None:
    df = load_state_data()

    st.title('🧪 Listeria Soil Risk Estimator')
    st.caption(
        'Prototype decision-support app for growers, regulators, and policy users. '
        'It combines the reported state-level prevalence/concentration table with a transparent '
        'scenario engine based on the strongest risk drivers from your project.'
    )

    with st.expander('Important note on scientific use', expanded=True):
        st.markdown(
            """
            - **What this app does now:** uses the uploaded state-level prevalence/concentration estimates and the reported directional drivers of risk.
            - **What this app does not yet do:** run the original fitted HistGradientBoosting or LightGBM model directly.
            - **Best next upgrade:** export the final preprocessor + trained model as a serialized artifact and connect it here. The UI can stay almost exactly the same.
            """
        )

    with st.sidebar:
        st.header('Scenario inputs')
        state = st.selectbox('State', df['state'].tolist(), index=df['state'].tolist().index('Iowa') if 'Iowa' in df['state'].tolist() else 0)
        moisture_level = st.select_slider('Soil moisture level', options=['Low', 'Medium', 'High'], value='Medium')
        near_water = st.checkbox('Near major river, lake, floodplain, or wetland', value=True)

        st.subheader('Land use')
        cropland_pct = st.slider('Cropland (%)', 0, 100, 20)
        pasture_pct = st.slider('Pasture (%)', 0, 100, 10)
        shrubland_pct = st.slider('Shrubland (%)', 0, 100, 5)

        st.subheader('Weather / environment')
        rh_max = st.slider('Maximum relative humidity (%)', 20, 100, 85)
        ppt_sum_mm = st.slider('Recent precipitation total (mm)', 0, 100, 10)
        temp_mean_c = st.slider('Mean temperature (°C)', -10, 40, 18)
        temp_max_c = st.slider('Maximum temperature (°C)', -5, 50, 30)
        wind_mean_ms = st.slider('Mean wind speed (m/s)', 0.0, 15.0, 3.0, 0.5)
        elevation_m = st.slider('Elevation (m)', 0, 4000, 300)
        longitude = st.slider('Longitude', -125.0, -66.0, -93.0, 0.5)

    row = df.loc[df['state'] == state].iloc[0]
    inputs = {
        'state': state,
        'moisture_level': moisture_level,
        'near_water': near_water,
        'cropland_pct': cropland_pct,
        'pasture_pct': pasture_pct,
        'shrubland_pct': shrubland_pct,
        'rh_max': rh_max,
        'ppt_sum_mm': ppt_sum_mm,
        'temp_mean_c': temp_mean_c,
        'temp_max_c': temp_max_c,
        'wind_mean_ms': wind_mean_ms,
        'elevation_m': elevation_m,
        'longitude': longitude,
    }
    scenario = estimate_risk(
        baseline_prevalence=float(row['prevalence']),
        p_upper=float(row['p_upper']),
        baseline_mean=float(row['cs_li_mean_cfu_g']),
        baseline_q95=float(row['sim_q95_cfu_g']),
        inputs=inputs,
    )

    left, right = st.columns([1.1, 1.2], gap='large')

    with left:
        st.subheader('Reported state baseline')
        c1, c2, c3 = st.columns(3)
        c1.metric('Prevalence', f"{row['prevalence']:.3f}")
        c2.metric('Mean concentration (CFU/g)', f"{row['cs_li_mean_cfu_g']:.3f}")
        c3.metric('95th percentile (CFU/g)', f"{row['sim_q95_cfu_g']:.3f}")

        st.dataframe(
            pd.DataFrame(
                {
                    'Field': ['State', 'Sample size (n)', 'Upper prevalence bound', '95% CI'],
                    'Value': [row['state'], int(row['n']), f"{row['p_upper']:.3f}", row['ci_95']],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader('Prototype scenario output')
        c4, c5, c6 = st.columns(3)
        c4.metric('Predicted probability of positive', f"{scenario['adjusted_probability']:.3f}")
        c5.metric('Estimated concentration (CFU/g)', f"{scenario['adjusted_mean']:.3f}")
        c6.metric('Estimated 95th percentile (CFU/g)', f"{scenario['adjusted_q95']:.3f}")

        st.markdown(
            f"**Risk band:** {scenario['risk_band']}  \\\n"
            f"**Multiplier vs. state baseline concentration:** {scenario['multiplier']:.2f}×"
        )

        st.download_button(
            'Download scenario as JSON',
            data=download_payload(row, scenario, inputs),
            file_name=f"listeria_scenario_{state.lower().replace(' ', '_')}.json",
            mime='application/json',
        )

    with right:
        st.subheader('Which inputs moved the estimate?')
        st.bar_chart(scenario['contributions'].set_index('Factor'))

        st.subheader('How to read this')
        st.markdown(
            """
            Positive bars push the estimate upward; negative bars pull it downward.
            This is designed to be **transparent and demo-friendly**, so judges and end users can see *why*
            the estimate changes.
            """
        )

        st.subheader('How to upgrade this to the real competition model')
        st.code(
            """
# 1) Export from your training pipeline:
#    - preprocessor.pkl or feature_order.json
#    - final_model.pkl (HistGB or LightGBM)
#    - optional concentration calibration object
# 2) Load those objects in app.py
# 3) Replace estimate_risk(...) with model.predict_proba(...)
# 4) Keep the same UI and reported state baseline panel
            """.strip(),
            language='python',
        )

    st.divider()
    st.subheader('State-level table used by the app')
    st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == '__main__':
    main()
