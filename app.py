from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier

APP_DIR = Path(__file__).resolve().parent

STATE_DATA_CANDIDATES = [
    APP_DIR / "data" / "state_estimates.csv",
    APP_DIR / "state_estimates.csv",
]
CLEAN_DATA_CANDIDATES = [
    APP_DIR / "data" / "processed" / "ListeriaSoil_clean_binary.csv",
    APP_DIR / "data" / "ListeriaSoil_clean_binary.csv",
    APP_DIR / "ListeriaSoil_clean_binary.csv",
]

REPORTED_BENCHMARK = {
    "Model": "HistGradientBoosting",
    "Threshold": 0.198421,
    "Recall": 0.919355,
    "Specificity": 0.888889,
    "Balanced Accuracy": 0.904122,
    "ROC AUC": 0.958013,
    "Avg Precision": 0.962916,
    "TP": 57,
    "TN": 56,
    "FP": 7,
    "FN": 5,
}

STATE_CENTROIDS = {
    "Alabama": (32.806671, -86.791130),
    "Arizona": (33.729759, -111.431221),
    "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564),
    "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371),
    "Delaware": (39.318523, -75.507141),
    "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074),
    "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353),
    "Nevada": (38.313515, -117.055374),
    "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755),
    "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434),
    "Washington": (47.400902, -121.490494),
    "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
}

STATE_ABBR = {
    "Alabama":"AL","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT",
    "Delaware":"DE","Florida":"FL","Georgia":"GA","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA",
    "Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA",
    "Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nevada":"NV",
    "New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH",
    "Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN",
    "Texas":"TX","Utah":"UT","Washington":"WA","Wisconsin":"WI","Wyoming":"WY"
}

CORE_FEATURES = [
    "Moisture",
    "Elevation (m)",
    "Precipitation (mm)",
    "Max temperature (℃ )",
    "Min temperature (℃ )",
    "Wind speed (m/s)",
    "Open water (%)",
    "Shrubland (%)",
    "Cropland (%)",
    "Pasture (%)",
    "Wetland (%)",
]

SOIL_FEATURES = [
    "pH",
    "Total nitrogen (%)",
    "Total carbon (%)",
    "Organic matter (%)",
    "Aluminum (mg/Kg)",
    "Calcium (mg/Kg)",
    "Copper (mg/Kg)",
    "Iron (mg/Kg)",
    "Potassium (mg/Kg)",
    "Magnesium (mg/Kg)",
    "Manganese (mg/Kg)",
    "Molybdenum (mg/Kg)",
    "Sodium (mg/Kg)",
    "Phosphorus (mg/Kg)",
    "Sulfur (mg/Kg)",
    "Zinc (mg/Kg)",
]

LAND_FEATURES = [
    "Developed open space (< 20% Impervious Cover) (%)",
    "Developed open space (> 20% Impervious Cover) (%)",
    "Barren (%)",
    "Forest (%)",
    "Grassland (%)",
]

st.set_page_config(page_title="Listeria Soil Risk Estimator v1", page_icon="🧪", layout="wide")


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


@st.cache_data
def load_state_data() -> pd.DataFrame:
    p = first_existing(STATE_DATA_CANDIDATES)
    if p is None:
        raise FileNotFoundError("state_estimates.csv not found.")
    df = pd.read_csv(p)
    for col in ["n", "prevalence", "p_upper", "cs_li_mean_cfu_g", "sim_q95_cfu_g"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["abbr"] = df["state"].map(STATE_ABBR)
    return df.sort_values("state").reset_index(drop=True)


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    p = first_existing(CLEAN_DATA_CANDIDATES)
    if p is None:
        raise FileNotFoundError(
            "ListeriaSoil_clean_binary.csv not found. Put it in data/processed/ or data/."
        )
    return pd.read_csv(p)


@st.cache_data
def get_feature_summary() -> dict:
    clean = load_clean_data()
    summary = {}
    for col in clean.columns:
        if col in ["Number of Listeria isolates obtained", "label"]:
            continue
        s = pd.to_numeric(clean[col], errors="coerce")
        summary[col] = {
            "min": float(np.nanmin(s)),
            "max": float(np.nanmax(s)),
            "median": float(np.nanmedian(s)),
        }
    return summary


@st.cache_resource
def train_competition_model():
    clean = load_clean_data()
    feature_cols = [c for c in clean.columns if c not in ["Number of Listeria isolates obtained", "label"]]
    X = clean[feature_cols].copy()
    y = clean["label"].astype(int)

    model = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.05,
        max_iter=400,
        min_samples_leaf=5,
        l2_regularization=1.0,
        max_bins=255,
        random_state=142,
    )
    model.fit(X, y)
    return model, feature_cols


def risk_band(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Moderate"
    return "Lower"


def concentration_from_probability(prob: float, state_row: pd.Series) -> Tuple[float, float, float]:
    baseline_p = float(state_row["prevalence"])
    anchor = baseline_p if baseline_p > 0 else max(0.02, float(state_row["p_upper"]) / 2.0)
    multiplier = float(np.clip(prob / max(anchor, 0.05), 0.35, 3.0))
    mean_c = float(state_row["cs_li_mean_cfu_g"]) * multiplier
    q95_c = float(state_row["sim_q95_cfu_g"]) * multiplier
    return mean_c, q95_c, multiplier


def make_slider(st_obj, label: str, summary: dict) -> float:
    info = summary[label]
    min_v = float(info["min"])
    max_v = float(info["max"])
    median_v = float(info["median"])
    step = max((max_v - min_v) / 200.0, 0.01)
    return st_obj.slider(label, min_value=min_v, max_value=max_v, value=median_v, step=step)


def build_input_row(feature_cols: list[str], state: str, use_exact_coords: bool, lat_value: float | None, lon_value: float | None, overrides: Dict[str, float], summary: dict) -> pd.DataFrame:
    row = {c: float(summary[c]["median"]) for c in feature_cols}
    default_lat, default_lon = STATE_CENTROIDS.get(state, (39.8283, -98.5795))
    row["Latitude"] = float(lat_value if use_exact_coords and lat_value is not None else default_lat)
    row["Longitude"] = float(lon_value if use_exact_coords and lon_value is not None else default_lon)
    for k, v in overrides.items():
        if k in row:
            row[k] = float(v)
    return pd.DataFrame([row], columns=feature_cols)


def build_state_map(state_df: pd.DataFrame, selected_state: str):
    fig = px.choropleth(
        state_df,
        locations="abbr",
        locationmode="USA-states",
        color="prevalence",
        scope="usa",
        hover_name="state",
        hover_data={
            "prevalence": ":.3f",
            "n": True,
            "cs_li_mean_cfu_g": ":.3f",
            "sim_q95_cfu_g": ":.3f",
            "abbr": False,
        },
        color_continuous_scale="YlOrRd",
        labels={
            "prevalence": "Baseline prevalence",
            "cs_li_mean_cfu_g": "Mean CFU/g",
            "sim_q95_cfu_g": "95th percentile CFU/g",
            "n": "Samples",
        },
    )
    sel = state_df.loc[state_df["state"] == selected_state]
    if not sel.empty:
        fig.add_scattergeo(
            locations=sel["abbr"],
            locationmode="USA-states",
            mode="markers",
            marker=dict(size=18, symbol="circle-open", line=dict(width=3, color="#003366")),
            name="Selected state",
            hoverinfo="skip",
        )
    fig.update_layout(margin=dict(l=0, r=0, t=25, b=0), coloraxis_colorbar_title="Prevalence")
    return fig


def main():
    st.title("🧪 Listeria Soil Risk Estimator v1")
    st.caption(
        "Competition-style v1: deployable HistGradientBoosting probability model + state-anchored concentration estimates."
    )

    state_df = load_state_data()
    summary = get_feature_summary()
    model, feature_cols = train_competition_model()

    with st.sidebar:
        st.header("Location and scenario")
        states = state_df["state"].tolist()
        default_idx = states.index("Iowa") if "Iowa" in states else 0
        state = st.selectbox("State", states, index=default_idx)

        use_exact_coords = st.checkbox("Specify exact latitude/longitude", value=False)
        centroid_lat, centroid_lon = STATE_CENTROIDS.get(state, (39.8283, -98.5795))
        if use_exact_coords:
            lat_value = st.number_input("Latitude", min_value=24.0, max_value=49.5, value=float(centroid_lat), step=0.01)
            lon_value = st.number_input("Longitude", min_value=-125.0, max_value=-66.0, value=float(centroid_lon), step=0.01)
        else:
            lat_value = None
            lon_value = None
            st.caption(f"Using {state} centroid: {centroid_lat:.2f}, {centroid_lon:.2f}")

        st.subheader("Core predictors")
        overrides = {}
        for label in CORE_FEATURES:
            overrides[label] = make_slider(st, label, summary)

        with st.expander("Advanced soil chemistry"):
            for label in SOIL_FEATURES:
                overrides[label] = make_slider(st, label, summary)

        with st.expander("Advanced land cover"):
            for label in LAND_FEATURES:
                overrides[label] = make_slider(st, label, summary)

    input_df = build_input_row(feature_cols, state, use_exact_coords, lat_value, lon_value, overrides, summary)
    prob = float(model.predict_proba(input_df)[0, 1])
    state_row = state_df.loc[state_df["state"] == state].iloc[0]
    mean_c, q95_c, ratio = concentration_from_probability(prob, state_row)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted probability", f"{prob:.3f}")
    c2.metric("Risk band", risk_band(prob))
    c3.metric("Estimated mean CFU/g", f"{mean_c:.3f}")
    c4.metric("Estimated 95th percentile CFU/g", f"{q95_c:.3f}")

    left, right = st.columns([1.05, 1.25], gap="large")
    with left:
        st.markdown("#### Model and interpretation")
        st.markdown(
            f"""
- **Classification model:** HistGradientBoosting  
- **Competition threshold:** `{REPORTED_BENCHMARK['Threshold']:.6f}`  
- **Selected state baseline prevalence:** `{float(state_row['prevalence']):.3f}`  
- **Scenario concentration multiplier:** `{ratio:.2f}×`
            """
        )
        bench = pd.DataFrame([REPORTED_BENCHMARK])
        st.dataframe(bench, use_container_width=True, hide_index=True)

        baseline = pd.DataFrame(
            {
                "Field": [
                    "State",
                    "Sample size (n)",
                    "Baseline prevalence",
                    "Upper prevalence bound",
                    "95% CI",
                    "Baseline mean concentration (CFU/g)",
                    "Baseline 95th percentile (CFU/g)",
                ],
                "Value": [
                    state_row["state"],
                    int(state_row["n"]),
                    f"{float(state_row['prevalence']):.3f}",
                    f"{float(state_row['p_upper']):.3f}",
                    state_row["ci_95"],
                    f"{float(state_row['cs_li_mean_cfu_g']):.3f}",
                    f"{float(state_row['sim_q95_cfu_g']):.3f}",
                ],
            }
        )
        st.markdown("#### Selected state baseline")
        st.dataframe(baseline, use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### U.S. baseline map")
        st.plotly_chart(build_state_map(state_df, state), use_container_width=True)

    st.markdown("#### Input profile vs training medians")
    compare_features = ["Latitude", "Longitude"] + CORE_FEATURES
    compare = pd.DataFrame(
        {
            "Feature": compare_features,
            "Scenario input": [float(input_df.iloc[0][f]) for f in compare_features],
            "Training median": [float(summary[f]["median"]) for f in compare_features],
        }
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(x=compare["Feature"], y=compare["Scenario input"], name="Scenario"))
    fig.add_trace(go.Bar(x=compare["Feature"], y=compare["Training median"], name="Training median"))
    fig.update_layout(barmode="group", margin=dict(l=0, r=0, t=20, b=0), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Probability is generated by the deployed HistGradientBoosting model. Concentration is estimated by scaling the state baseline concentration using the scenario-to-baseline probability ratio."
    )

    payload = {
        "state": state,
        "prediction": {
            "probability_positive": prob,
            "risk_band": risk_band(prob),
            "estimated_mean_cfu_g": mean_c,
            "estimated_q95_cfu_g": q95_c,
            "concentration_multiplier": ratio,
        },
        "benchmark": REPORTED_BENCHMARK,
        "inputs": input_df.iloc[0].to_dict(),
    }
    st.download_button(
        "Download current scenario JSON",
        data=json.dumps(payload, indent=2).encode("utf-8"),
        file_name=f"listeria_v1_{state.lower().replace(' ', '_')}.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
