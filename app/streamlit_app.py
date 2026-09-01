"""
Pearls AQI Predictor -- Streamlit dashboard.

Run locally with:

    streamlit run app/streamlit_app.py

Loads the latest features + registered models from whichever backend is
configured (local Parquet/joblib by default, Hopsworks if configured -- see
.env.example), renders the 3-day forecast per city, a historical trend
chart, SHAP-based explanations, hazard alerts, and model performance.

If no data has been backfilled/trained yet, the sidebar offers a one-click
"first-time setup" that runs the backfill + training pipelines directly
from the app, so a fresh clone becomes a working dashboard in a couple of
minutes with zero command-line use.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aqi_predictor.alerts.aqi_alerts import check_forecast_for_alerts
from aqi_predictor.config import (
    AQI_CATEGORIES,
    AQI_DATA_PROVIDER,
    DEFAULT_CITIES,
    FEATURE_STORE_BACKEND,
    FORECAST_HORIZON_DAYS,
    CITY_BY_KEY,
    categorize_aqi,
)
from aqi_predictor.explainability.shap_explainer import (
    explain_single_prediction,
    global_feature_importance,
)
from aqi_predictor.features.engineering import FEATURE_COLUMNS, training_frame
from aqi_predictor.features.feature_store import get_feature_store
from aqi_predictor.models.forecaster import CityForecast, forecast_many
from aqi_predictor.models.registry import get_model_registry
from aqi_predictor.pipelines.backfill_pipeline import run_backfill_pipeline
from aqi_predictor.pipelines.training_pipeline import run_training_pipeline

# ---------------------------------------------------------------------------
# Page setup + visual identity
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="\U0001F32B\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --ink: #131A2A;
    --paper: #F7F8FA;
    --card: #FFFFFF;
    --line: #E4E7EC;
    --navy: #0B1220;
    --navy-2: #16213A;
    --accent: #2E5EAA;
}

html, body, [class*="css"]  { font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; }
h1, h2, h3, .hero-title { font-family: 'Space Grotesk', ui-sans-serif, sans-serif; }
.mono, .aqi-number { font-family: 'IBM Plex Mono', ui-monospace, monospace; }

.hero {
    background: linear-gradient(120deg, var(--navy) 0%, var(--navy-2) 100%);
    color: #F2F4F8;
    padding: 28px 32px;
    border-radius: 10px;
    margin-bottom: 22px;
}
.hero-title { font-size: 1.9rem; font-weight: 700; margin: 0; letter-spacing: -0.01em; }
.hero-sub { color: #AEB8CC; font-size: 0.95rem; margin-top: 6px; }
.hero-meta { color: #7C8AA8; font-size: 0.8rem; margin-top: 14px; }

.city-strip { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 6px; margin-bottom: 18px; }
.city-chip {
    flex: 0 0 auto; min-width: 148px; background: var(--card); border: 1px solid var(--line);
    border-left: 5px solid var(--chip-color, #999); border-radius: 6px; padding: 10px 14px;
}
.city-chip .name { font-size: 0.8rem; color: #5B657A; font-weight: 500; }
.city-chip .value { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: var(--ink); }
.city-chip .cat { font-size: 0.72rem; font-weight: 600; color: var(--chip-color, #999); }

.aqi-card {
    background: var(--card); border: 1px solid var(--line); border-left: 6px solid var(--chip-color, #999);
    border-radius: 6px; padding: 16px 18px; height: 100%;
}
.aqi-card .label { font-size: 0.78rem; color: #5B657A; font-weight: 500; text-transform: none; }
.aqi-card .aqi-number { font-size: 2.1rem; font-weight: 600; color: var(--ink); line-height: 1.1; margin-top: 4px; }
.aqi-card .cat-label { font-size: 0.82rem; font-weight: 600; color: var(--chip-color, #999); margin-top: 2px; }
.aqi-card .date-label { font-size: 0.72rem; color: #8A93A6; margin-top: 8px; }

.alert-banner {
    background: #FCEBEA; border: 1px solid #E0473E; color: #7A1710;
    border-radius: 6px; padding: 12px 16px; margin-bottom: 14px; font-size: 0.88rem;
}
.empty-state {
    background: var(--card); border: 1px dashed var(--line); border-radius: 8px;
    padding: 36px; text-align: center; color: #5B657A;
}
.section-label {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em; color: #5B657A;
    margin-bottom: 6px; margin-top: 4px;
}
footer, #MainMenu { visibility: hidden; }
.app-footer { color: #8A93A6; font-size: 0.78rem; margin-top: 28px; border-top: 1px solid var(--line); padding-top: 14px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data access (keyed on simple, hashable args -- never on the
# FeatureStore/ModelRegistry objects themselves)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_feature_table(city_keys: tuple[str, ...]) -> pd.DataFrame:
    return get_feature_store().read_features(city_keys=list(city_keys))


@st.cache_data(ttl=180, show_spinner=False)
def load_forecasts(city_keys: tuple[str, ...]):
    store = get_feature_store()
    registry = get_model_registry()
    return forecast_many(list(city_keys), store, registry)


@st.cache_data(ttl=600, show_spinner=False)
def load_training_history() -> pd.DataFrame:
    registry = get_model_registry()
    history_path = getattr(registry, "directory", None)
    if history_path is None:
        return pd.DataFrame()
    history_file = history_path / "training_history.jsonl"
    if not history_file.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in history_file.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["rmse"] = df["metrics"].apply(lambda m: m.get("rmse"))
    df["mae"] = df["metrics"].apply(lambda m: m.get("mae"))
    df["r2"] = df["metrics"].apply(lambda m: m.get("r2"))
    # most recent registration per horizon
    return df.sort_values("trained_at").groupby("horizon_days").tail(1).sort_values("horizon_days")


@st.cache_data(ttl=600, show_spinner=False)
def load_global_importance(horizon: int) -> pd.DataFrame | None:
    registry = get_model_registry()
    directory = getattr(registry, "directory", None)
    if directory is not None:
        path = directory / f"feature_importance_h{horizon}d.csv"
        if path.exists():
            return pd.read_csv(path)
    try:
        model, meta = registry.load_model(horizon)
        feats = get_feature_store().read_features()
        background = training_frame(feats, FEATURE_COLUMNS, [f"target_{horizon}d"])
        if background.empty:
            return None
        return global_feature_importance(model, meta["model_type"], FEATURE_COLUMNS, background)
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def compute_shap_explanation(city_key: str, horizon: int, as_of: str) -> pd.DataFrame | None:
    try:
        registry = get_model_registry()
        store = get_feature_store()
        model, meta = registry.load_model(horizon)
        feats = store.read_features()
        background = training_frame(feats, FEATURE_COLUMNS, [f"target_{horizon}d"])
        latest = store.read_features(city_keys=[city_key])
        if background.empty or latest.empty:
            return None
        row = latest.loc[latest["date"].idxmax()]
        return explain_single_prediction(model, meta["model_type"], FEATURE_COLUMNS, background, row)
    except Exception:
        return None


def clear_all_caches() -> None:
    load_feature_table.clear()
    load_forecasts.clear()
    load_training_history.clear()
    load_global_importance.clear()
    compute_shap_explanation.clear()


# ---------------------------------------------------------------------------
# Small render helpers
# ---------------------------------------------------------------------------
def aqi_card_html(label: str, value: float, category, date_label: str) -> str:
    return f"""
    <div class="aqi-card" style="--chip-color:{category.color};">
        <div class="label">{label}</div>
        <div class="aqi-number">{value:.0f}</div>
        <div class="cat-label">{category.name}</div>
        <div class="date-label">{date_label}</div>
    </div>
    """


def city_chip_html(city_name: str, value: float, category) -> str:
    return f"""
    <div class="city-chip" style="--chip-color:{category.color};">
        <div class="name">{city_name}</div>
        <div class="value">{value:.0f}</div>
        <div class="cat">{category.name}</div>
    </div>
    """


def render_legend() -> None:
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px;'
        f'font-size:0.78rem;color:#5B657A;"><span style="width:10px;height:10px;border-radius:2px;'
        f'background:{c.color};display:inline-block;"></span>{c.name} ({c.low}-{c.high})</span>'
        for c in AQI_CATEGORIES
    )
    st.markdown(f'<div style="margin:6px 0 18px 0;">{chips}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### \U0001F32B\uFE0F Pearls AQI Predictor")
    st.caption("3-day Air Quality Index forecasts, 100% serverless.")

    all_city_keys = [c.key for c in DEFAULT_CITIES]
    selected_keys = st.multiselect(
        "Cities to track",
        options=all_city_keys,
        default=all_city_keys,
        format_func=lambda k: CITY_BY_KEY[k].name,
    )
    if not selected_keys:
        selected_keys = all_city_keys

    st.divider()
    st.markdown("**Data provider**")
    st.caption(f"`{AQI_DATA_PROVIDER}` (weather + AQI)")
    st.markdown("**Feature store / registry**")
    st.caption(f"`{FEATURE_STORE_BACKEND}`")

    st.divider()
    with st.expander("First-time setup / refresh data", expanded=False):
        st.caption(
            "No data yet, or want to add more history? This runs the backfill "
            "and training pipelines directly (may take a few minutes and calls "
            "the live weather/AQI API)."
        )
        lookback = st.number_input("Days of history to backfill", min_value=30, max_value=365,
                                    value=120, step=10)
        setup_cities = st.multiselect(
            "Cities to set up", options=all_city_keys, default=selected_keys,
            format_func=lambda k: CITY_BY_KEY[k].name, key="setup_cities",
        )
        if st.button("Run backfill + training now", type="primary", use_container_width=True):
            cities_to_run = [CITY_BY_KEY[k] for k in setup_cities] or DEFAULT_CITIES
            with st.spinner(f"Backfilling {lookback} days for {len(cities_to_run)} cities..."):
                run_backfill_pipeline(cities=cities_to_run, lookback_days=int(lookback))
            with st.spinner("Training models..."):
                try:
                    run_training_pipeline(cities=cities_to_run)
                    st.success("Setup complete!")
                except RuntimeError as exc:
                    st.warning(f"Backfill done, but training couldn't complete yet: {exc}")
            clear_all_caches()
            st.rerun()

    st.divider()
    st.caption(
        "Data: Open-Meteo (CAMS atmospheric composition + ERA5/forecast weather models). "
        "AQI categories follow the US EPA scale."
    )

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
now_str = pd.Timestamp.now().strftime("%A, %d %B %Y - %H:%M")
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">Pearls AQI Predictor</div>
        <div class="hero-sub">3-day Air Quality Index forecasts for {len(selected_keys)} tracked
        {'city' if len(selected_keys) == 1 else 'cities'}, updated hourly.</div>
        <div class="hero-meta">Dashboard rendered {now_str}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
selected_keys_t = tuple(sorted(selected_keys))
feature_table = load_feature_table(selected_keys_t)

if feature_table.empty:
    st.markdown(
        """
        <div class="empty-state">
            <h3>No data yet</h3>
            <p>This dashboard reads from the local feature store, which is currently empty.</p>
            <p>Open <b>"First-time setup / refresh data"</b> in the sidebar and click
            <b>"Run backfill + training now"</b> - or run from the command line:</p>
            <p class="mono" style="text-align:left; display:inline-block; background:#F1F3F6;
               padding:10px 14px; border-radius:6px;">
               python scripts/run_backfill.py<br>
               python scripts/run_training_pipeline.py
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

forecasts = load_forecasts(selected_keys_t)

# ---------------------------------------------------------------------------
# City overview strip
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">CURRENT CONDITIONS</div>', unsafe_allow_html=True)
chips = []
for key in selected_keys:
    result = forecasts.get(key)
    if isinstance(result, CityForecast):
        cat = categorize_aqi(result.latest_observed_aqi)
        chips.append(city_chip_html(CITY_BY_KEY[key].name, result.latest_observed_aqi, cat))
st.markdown(f'<div class="city-strip">{"".join(chips)}</div>', unsafe_allow_html=True)
render_legend()

# ---------------------------------------------------------------------------
# City detail view
# ---------------------------------------------------------------------------
available_keys = [k for k in selected_keys if isinstance(forecasts.get(k), CityForecast)]
if not available_keys:
    st.warning(
        "No forecasts are available yet for the selected cities - models may still need "
        "training, or these cities need more backfilled history. See the sidebar setup panel."
    )
    st.stop()

detail_key = st.selectbox(
    "City detail", options=available_keys, format_func=lambda k: CITY_BY_KEY[k].name
)
forecast: CityForecast = forecasts[detail_key]
alerts = check_forecast_for_alerts(forecast)

if alerts:
    for alert in alerts:
        st.markdown(f'<div class="alert-banner">\u26A0\uFE0F {alert.message}</div>',
                     unsafe_allow_html=True)

cols = st.columns(1 + FORECAST_HORIZON_DAYS)
with cols[0]:
    st.markdown(
        aqi_card_html("Current AQI", forecast.latest_observed_aqi,
                      categorize_aqi(forecast.latest_observed_aqi),
                      f"as of {forecast.as_of_date:%b %d, %Y}"),
        unsafe_allow_html=True,
    )
for i, point in enumerate(forecast.points):
    with cols[i + 1]:
        st.markdown(
            aqi_card_html(f"Day +{point.horizon_days}", point.predicted_aqi, point.category,
                          f"{point.target_date:%a, %b %d}"),
            unsafe_allow_html=True,
        )

st.write("")
left, right = st.columns([3, 2])

# --- Historical trend + forecast chart -------------------------------------
with left:
    st.markdown('<div class="section-label">30-DAY TREND + FORECAST</div>', unsafe_allow_html=True)
    city_history = feature_table[feature_table["city_key"] == detail_key].sort_values("date")
    cutoff = forecast.as_of_date - timedelta(days=30)
    city_history = city_history[city_history["date"] >= cutoff]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=city_history["date"], y=city_history["us_aqi_mean"],
        mode="lines+markers", name="Observed", line=dict(color="#2E5EAA", width=2),
        marker=dict(size=5),
    ))
    forecast_x = [forecast.as_of_date] + [p.target_date for p in forecast.points]
    forecast_y = [forecast.latest_observed_aqi] + [p.predicted_aqi for p in forecast.points]
    fig.add_trace(go.Scatter(
        x=forecast_x, y=forecast_y, mode="lines+markers", name="Forecast",
        line=dict(color="#E0473E", width=2, dash="dot"), marker=dict(size=7, symbol="diamond"),
    ))
    for boundary in (50, 100, 150, 200, 300):
        fig.add_hline(y=boundary, line_dash="dot", line_color="#D8DCE3", line_width=1)
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="US AQI", xaxis_title=None,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

# --- SHAP: why this forecast -------------------------------------------
with right:
    st.markdown('<div class="section-label">WHY THIS FORECAST (SHAP, DAY +1)</div>',
                unsafe_allow_html=True)
    with st.spinner("Computing explanation..."):
        explanation = compute_shap_explanation(detail_key, 1, str(forecast.as_of_date))
    if explanation is None or explanation.empty:
        st.caption("Explanation unavailable for this city/model yet.")
    else:
        explanation = explanation.sort_values("shap_value")
        colors = ["#E0473E" if v > 0 else "#2E5EAA" for v in explanation["shap_value"]]
        fig2 = go.Figure(go.Bar(
            x=explanation["shap_value"], y=explanation["feature"], orientation="h",
            marker_color=colors,
        ))
        fig2.update_layout(
            height=360, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Impact on predicted AQI", plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Red = pushes tomorrow's AQI higher, blue = pushes it lower.")

# ---------------------------------------------------------------------------
# Model performance + global feature importance
# ---------------------------------------------------------------------------
st.write("")
perf_col, importance_col = st.columns([2, 3])

with perf_col:
    st.markdown('<div class="section-label">MODEL PERFORMANCE (HELD-OUT TEST SET)</div>',
                unsafe_allow_html=True)
    history = load_training_history()
    if history.empty:
        st.caption("No training history yet.")
    else:
        display_df = history[["horizon_days", "model_type", "rmse", "mae", "r2"]].rename(
            columns={"horizon_days": "Horizon (days)", "model_type": "Model",
                     "rmse": "RMSE", "mae": "MAE", "r2": "R2"}
        )
        st.dataframe(display_df, hide_index=True, use_container_width=True)

with importance_col:
    st.markdown('<div class="section-label">GLOBAL FEATURE IMPORTANCE (DAY +1 MODEL)</div>',
                unsafe_allow_html=True)
    importance = load_global_importance(1)
    if importance is None or importance.empty:
        st.caption("Feature importance not available yet.")
    else:
        top = importance.head(10).sort_values("mean_abs_shap")
        fig3 = go.Figure(go.Bar(
            x=top["mean_abs_shap"], y=top["feature"], orientation="h",
            marker_color="#2E5EAA",
        ))
        fig3.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Mean |SHAP value|", plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        Pearls AQI Predictor - weather + air-quality data from
        <a href="https://open-meteo.com" target="_blank">Open-Meteo</a>
        (CAMS atmospheric composition, ERA5 &amp; forecast weather models).
        AQI categories follow the US EPA scale. Forecasts are estimates from a
        statistical/ML model, not an official air-quality advisory - consult your
        local environmental agency for health guidance.
    </div>
    """,
    unsafe_allow_html=True,
)
