# Pearls AQI Predictor — Project Report

## 1. Executive summary

This project delivers an end-to-end, 100%-serverless system that forecasts
the US Air Quality Index (AQI) 1, 2, and 3 days ahead for a configurable
set of cities. It covers every stage the brief asked for: a feature
pipeline, historical backfill, a training pipeline that compares several
model families and registers the best one per horizon, hourly/daily
automation via GitHub Actions, an interactive dashboard with SHAP-based
explanations and hazardous-AQI alerts, and this report.

The codebase is organised as an installable Python package
(`src/aqi_predictor/`) with a full offline pytest suite (63 tests), rather
than a collection of notebooks, so it can be run, tested, and extended the
way a production service would be.

## 2. Goals, restated

From the project brief:

1. Feature pipeline: fetch raw weather + pollutant data, compute features
   (incl. time-based + derived features like AQI change rate), store in a
   feature store.
2. Backfill historical (features, targets) for training data.
3. Training pipeline: fetch from the feature store, train/evaluate several
   models (statistical → deep learning), store the best in a model
   registry.
4. Automate both pipelines (hourly / daily).
5. A web dashboard: load model + features, compute predictions, show them
   on a dashboard.
6. EDA, SHAP/LIME explanations, hazard alerts, a detailed report.

Every one of these has a corresponding, working implementation in this
repo — see the README's [project structure](README.md#project-structure)
for exactly where.

## 3. Data source

**Chosen provider: [Open-Meteo](https://open-meteo.com)** (Air Quality API
+ Historical Weather API + Forecast API), used via
`src/aqi_predictor/data/api_client.py::OpenMeteoClient`.

The brief names AQICN/OpenWeather as *examples* ("you may need to explore
other options too"). Open-Meteo was chosen because:

- **No signup or API key required** — the single biggest factor in making
  this a "clone and it just works" deliverable rather than one that's
  blocked on someone creating three different accounts first.
- Its Air Quality API returns a **properly computed US AQI** (`us_aqi`)
  directly, backed by the CAMS (Copernicus Atmosphere Monitoring Service)
  global composition model — no need to derive AQI from raw pollutant
  concentrations ourselves, and the value is comparable across cities and
  time.
- **Deep historical coverage**: the Historical Weather API (ERA5
  reanalysis) goes back to 1940; the Air Quality API's CAMS global data is
  available from August 2022 onward, which is ample for backfilling
  several months to a couple of years of daily training data.
- Generous free-tier rate limits (no realistic risk of hitting a wall from
  an hourly pipeline over a handful of cities).

**Important nuance handled explicitly in the code**: Open-Meteo's
`/v1/archive` endpoint (ERA5 reanalysis, used for *deep* historical
weather) is only updated with roughly a 5-day publication delay. Naively
using it for "yesterday" or "today" would silently return nothing useful.
`OpenMeteoClient.fetch_recent()` therefore uses the *forecast* API's
`past_days` parameter instead for the hourly feature pipeline (recent
data), while `fetch_historical()` uses the archive endpoint for deep
backfill — see the docstrings in `api_client.py` for the full reasoning.

**Secondary adapter**: `OpenWeatherClient` implements the same interface
against OpenWeather's free-tier Air Pollution / Weather APIs, for anyone
who wants to switch (`AQI_DATA_PROVIDER=openweather` + an API key). Since
OpenWeather's Air Pollution API returns raw pollutant concentrations
rather than a 0–500 AQI, `src/aqi_predictor/data/aqi_math.py` implements
the EPA breakpoint formula (2024-revised breakpoints) to derive a
comparable AQI value — clearly documented there as an *approximation*
(instantaneous readings against breakpoints defined for rolling averages),
since being upfront about that gap matters more than pretending it isn't
there.

## 4. Feature engineering

Implemented in `src/aqi_predictor/features/engineering.py`.

**Grain**: raw hourly rows are aggregated to **one row per city per day**
(`aggregate_hourly_to_daily`) — mean/max/min for AQI and temperature, mean
for other weather variables, sum for precipitation. Daily grain matches
how AQI forecasts are actually consumed ("what will the air be like
tomorrow"), keeps the feature/target tables small enough to inspect by
eye, and avoids over-fitting to hour-of-day noise that the calendar
features already capture at a coarser, more robust level.

**Time-based features**: `day_of_week`, `is_weekend`, `month`,
`day_of_year`, plus sine/cosine encodings of month and day-of-year so the
model sees seasonality as continuous and cyclical (December 31 and January
1 are one day apart, not 364).

**Derived features**:
- `aqi_change_rate` — day-over-day percentage change, exactly as the brief
  asked for, computed safely (guards against division by zero / missing
  history).
- Lag features at 1/2/3/7 days (`us_aqi_mean_lag_{1,2,3,7}d`).
- Rolling mean/std over 3- and 7-day windows, computed on `shift(1)` of the
  series so a day's own value is never part of its own rolling statistic
  (a common, easy-to-miss leakage bug).

**No cross-city leakage**: every lag/rolling operation is grouped by
`city_key` before shifting/rolling — verified explicitly in
`tests/test_feature_engineering.py::test_lags_and_rolling_do_not_leak_across_cities`.

**Targets**: `target_1d`, `target_2d`, `target_3d` — the AQI value N days
*ahead*, per city (`add_targets`), i.e. this is a genuine forecast, not a
same-day fit.

## 5. Modelling

Implemented in `src/aqi_predictor/models/trainer.py`.

### 5.1 Why three separate models instead of one multi-output model

A single model predicting `[target_1d, target_2d, target_3d]`
simultaneously (via `MultiOutputRegressor` or a 3-unit output layer) was
considered and rejected in favour of **three independently-trained
regressors, one per horizon**. Reasoning:

- Day+1 and day+3 have genuinely different error profiles (day+3 is
  strictly harder — see the persistence-baseline comparison below), so
  they benefit from being allowed to select *different* winning algorithms
  rather than one algorithm being forced to compromise across all three.
- It keeps model registry entries, evaluation, and the dashboard's
  per-horizon SHAP explanations simpler to reason about independently.
- The added training cost (3× fits instead of 1×) is negligible at this
  data scale and cadence (daily retraining, a handful of cities).

### 5.2 Candidates evaluated (statistical → deep learning, per the brief)

| Candidate | Type | Notes |
|---|---|---|
| `persistence` | naive baseline | "tomorrow = today's AQI" — a *reference*, never selected as the winner, but always shown in the leaderboard. If a real model can't beat this, that's the important finding. |
| `ridge` | statistical (linear) | `StandardScaler` + `Ridge`, scikit-learn |
| `random_forest` | classical ML ensemble | scikit-learn `RandomForestRegressor` |
| `xgboost` | gradient-boosted trees | skipped gracefully if `xgboost` isn't installed |
| `dense_nn` | deep learning | small Keras feed-forward network with early stopping; skipped gracefully if `tensorflow` isn't installed (it's an optional, heavy dependency — see `requirements.txt`) |

Every candidate is skipped (not a hard error) if its optional dependency is
missing, so the pipeline always completes with whatever's installed.

### 5.3 Train/test split

**Chronological**, not random k-fold (`trainer.time_based_split`): the most
recent `test_fraction` (default 20%) of *dates* become the test set, with
all cities' same-day rows kept together. Randomly shuffling rows before
splitting would leak future information into training (a model could learn
from a day that comes chronologically after some of its "test" days),
which would make offline metrics look better than real-world forecasting
performance ever could.

### 5.4 Evaluation metrics

- **RMSE** (root mean squared error) — the primary model-selection metric;
  penalises large misses more than small ones, which matters most right
  around the hazard-alert threshold.
- **MAE** (mean absolute error) — more interpretable ("average AQI points
  off"), less sensitive to outliers than RMSE.
- **R²** — how much of the AQI variance the model explains versus a
  constant-mean baseline; useful for sanity-checking that a model is
  actually learning structure, not just memorising noise.

All three are computed for every candidate and shown in the dashboard's
"Model performance" table and `training_history.jsonl`.

### 5.5 Validation run (synthetic data)

The pipelines were validated end-to-end against **synthetic** weather/AQI
data (seasonal + weekly + wind-dependent signal, 200 days, 3 cities) with
the real HTTP layer mocked to match Open-Meteo's documented response
schema exactly — this exercises every line of the actual pipeline code
without needing live network access during development.

| Horizon | Best model | RMSE | MAE | R² | Persistence RMSE (reference) |
|---|---|---|---|---|---|
| day+1 | ridge | 2.86 | 2.40 | 0.994 | 5.19 |
| day+2 | ridge | 3.43 | 2.86 | 0.992 | 8.94 |
| day+3 | ridge | 3.26 | 2.59 | 0.993 | 11.29 |

**These numbers describe the synthetic sanity-check, not real-world air
quality — they are reported here only to demonstrate the pipeline works
and to show what a leaderboard looks like.** Ridge wins here specifically
because the synthetic signal was constructed as a mostly-linear function of
its inputs; on real, messier data, `random_forest` or `xgboost` are
typically more competitive; that's exactly why all four run every time
rather than hard-coding a winner. To get real metrics, run:

```bash
python scripts/run_backfill.py --lookback-days 180
python scripts/run_training_pipeline.py
```
and read `data/local_store/model_registry/training_history.jsonl` or the
dashboard's "Model performance" panel.

## 6. Explainability (SHAP)

Implemented in `src/aqi_predictor/explainability/shap_explainer.py`.

- **Tree-based models** (`random_forest`, `xgboost`) use `shap.TreeExplainer`
  — exact and fast.
- **Every other model type** (`ridge`, `dense_nn`) falls back to the
  model-agnostic `shap.Explainer` driven off `model.predict`, at the cost
  of being slower — acceptable since explanations are computed for a
  handful of rows at a time (one per city), not the full training set.
- **Global importance** (`global_feature_importance`): mean absolute SHAP
  value per feature over a sample of held-out rows — computed once after
  training and cached to `feature_importance_h{N}d.csv` in the model
  registry directory, so the dashboard doesn't need to recompute it live.
- **Per-prediction explanation** (`explain_single_prediction`): "why is
  tomorrow's forecast for this city X?" — the top-K features by |SHAP
  value|, signed, rendered in the dashboard as a red/blue tornado chart.

LIME was not implemented alongside SHAP: SHAP's TreeExplainer gives exact,
fast attributions for the tree-based candidates (the most commonly-selected
model family in practice), and using one consistent explanation method
across every candidate — rather than switching frameworks per model type —
keeps the global vs. per-prediction numbers directly comparable.

## 7. Alerting

Implemented in `src/aqi_predictor/alerts/aqi_alerts.py`.

Any forecast day reaching **AQI ≥ 151 ("Unhealthy" or worse)** produces an
`Alert`. Alerts are:

- Rendered as banners in the dashboard for the selected city.
- Logged as warnings, and — if `SLACK_WEBHOOK_URL` is configured — posted
  to Slack automatically at the end of the daily training pipeline run
  (`training_pipeline.py::run_training_pipeline`), so a hazardous forecast
  reaches people even if nobody has the dashboard open that day.
- The feature pipeline additionally logs a lighter-weight "current
  conditions" warning if *today's observed* AQI is already hazardous
  (`feature_pipeline.py::_log_current_conditions`), independent of the
  forecast.

## 8. Feature store & model registry

Implemented behind common interfaces (`FeatureStore`, `ModelRegistry`) with
two backends each:

- **Local (default)**: a single Parquet file (features) and joblib files +
  JSON metadata (models) under `data/local_store/`. Zero setup, works
  offline, upserts are idempotent (keyed on `city_key` + `date` /
  `horizon`), and it's what makes `pip install && run` produce a working
  system with no external accounts.
- **Hopsworks (optional)**: the managed, "properly serverless" store
  suggested by the brief, activated via `FEATURE_STORE_BACKEND=hopsworks` +
  credentials (see `.env.example`). If Hopsworks is configured but
  unreachable (bad credentials, network issue, package not installed), the
  factory functions (`get_feature_store()`, `get_model_registry()`) log a
  clear warning and **fall back to the local backend** rather than crashing
  the pipeline — a deliberate resilience choice for scheduled automation,
  where a transient failure shouldn't take down the whole run.

## 9. Automation (CI/CD)

Three GitHub Actions workflows (`.github/workflows/`): hourly feature
pipeline, daily training pipeline, and a test workflow that runs pytest on
every push/PR.

**Persistence tradeoff, stated plainly**: on the default `local` backend, a
GitHub Actions runner's filesystem is thrown away after each job. To make
the hourly/daily automation actually accumulate history over time without
requiring a Hopsworks signup, the workflows commit the updated
`data/local_store/**` files straight back into the repository
(`git add -f ...; git commit; git push`, with `[skip ci]` to avoid
re-triggering the test workflow). This is a pragmatic, zero-cost pattern —
effectively using the git repo itself as a tiny append-mostly database —
and is clearly not what "serverless feature store" means in the strict,
managed-service sense. It is presented as the free default specifically so
the whole system works end-to-end the moment this repo is pushed to
GitHub, with `FEATURE_STORE_BACKEND=hopsworks` documented in the README as
the drop-in upgrade to a genuinely managed store once that trade-off
matters (e.g. multiple contributors, higher write frequency, or deploying
the dashboard somewhere with a read-only filesystem).

## 10. Known limitations & future work

- **Model performance numbers in this report are from synthetic data** (see
  §5.5) — real metrics require running the backfill + training pipelines
  against live data, which needs outbound internet access that wasn't
  available in the environment this project was built in.
- **OpenWeather AQI is an approximation** when that provider is selected
  (see §3) — Open-Meteo's native `us_aqi` is preferred and used by default
  for exactly this reason.
- **No online/streaming inference** — the dashboard reads the latest
  *stored* daily feature row; if the hourly pipeline hasn't run recently,
  forecasts are based on slightly stale features (surfaced to the user via
  the "as of" date on each forecast card).
- **Single global model per horizon**, not one model per city — with more
  cities and more history, a per-city or hierarchical model (or adding
  `city_key` as a categorical feature, which the current per-horizon
  models don't use) could capture city-specific dynamics better; this was
  left out to keep the registry/serving model simple for a first version.
- **Git-commit-based persistence doesn't scale indefinitely** — see §9;
  the Hopsworks backend is the documented upgrade path.
- **The `dense_nn` candidate requires `tensorflow`**, which is deliberately
  left out of the default install (`requirements.txt`) because it's a
  large, platform-sensitive dependency; install it explicitly
  (`pip install tensorflow`) to include a genuine deep-learning candidate
  in the leaderboard.

## 11. How to reproduce everything in this report

```bash
pip install -r requirements.txt && pip install -e .
python scripts/run_backfill.py --lookback-days 180
python scripts/run_training_pipeline.py
pytest -v
streamlit run app/streamlit_app.py
```
