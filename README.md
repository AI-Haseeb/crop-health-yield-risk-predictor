# Crop Health & Yield Risk Predictor

A full-stack AgriTech hackathon project that combines two trained ANN models with a FastAPI web dashboard.

The system recommends a crop from soil/weather inputs, predicts yield risk for that crop, maps the risk to a farmer decision, and optionally improves farmer advice using Groq LLM.

## Features

- Crop recommendation ANN
- Yield risk ANN
- Full ML inference pipeline
- FastAPI web app with Jinja templates
- Modern responsive AgriTech dashboard
- Terminal-style pipeline report for VS Code
- Optional Groq LLM farmer advice
- Safe fallback advice if Groq is unavailable

## Model Flow

1. User enters:
   - `N`
   - `P`
   - `K`
   - `temperature`
   - `humidity`
   - `ph`
   - `rainfall`
   - `area`
   - `year`
   - `pesticides_tonnes`

2. Model 1 predicts:
   - `recommended_crop`

3. Model 2 uses:
   - `area`
   - predicted crop as `item`
   - `year`
   - `rainfall` mapped to `average_rain_fall_mm_per_year`
   - `pesticides_tonnes`
   - `temperature` mapped to `avg_temp`

4. Final decision logic:
   - `LOW -> GO`
   - `MEDIUM -> CAUTION`
   - `HIGH -> HOLD`

## Required Artifacts

These files must exist:

```text
models/crop_recommendation_model.h5
models/scaler.pkl
models/label_encoder.pkl
models/yield_risk_ann.h5
models/risk_scaler.pkl
models/risk_encoder.pkl
data/raw/yield_df.csv
```

Risk feature support files:

```text
models/risk_model/crop_reference_stats.csv
models/risk_model/area_crop_reference_stats.csv
models/risk_model/yield_risk_features.json
```

## Strict Data Leakage Rule

The pipeline never uses these leakage columns as model inputs:

```text
hg_ha_yield
expected_yield
yield_ratio
yield_percentile
yield_risk_score
weather_risk_score
final_risk_score
yield_risk
```

## Project Structure

```text
crop-health-yield-risk-predictor/
  frontend/
    app.py
    templates/
      index.html
      result.html
    static/
      css/
        style.css

  models/
    crop_recommendation_model.h5
    scaler.pkl
    label_encoder.pkl
    yield_risk_ann.h5
    risk_scaler.pkl
    risk_encoder.pkl
    risk_model/
      crop_reference_stats.csv
      area_crop_reference_stats.csv
      yield_risk_features.json

  data/
    raw/
      yield_df.csv
      Crop_recommendation.csv
      rainfall.csv
      pesticides.csv
      temp.csv
      yield.csv

  reports/
  notebooks/
  project_structure/
    STRUCTURE.md

  pipeline.py
  test_pipeline.py
  requirements.txt
  README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Optional Groq LLM Setup

Groq is used only to improve the farmer advice text. It does not change model predictions, risk labels, or decisions.

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

`.env` is ignored by Git and should not be pushed.

If Groq fails or the key is missing, the app automatically uses the built-in fallback advice.

## Run Terminal Pipeline Test

```bash
python test_pipeline.py
```

This prints a clean VS Code terminal report:

```text
CROP HEALTH & YIELD RISK PREDICTION REPORT
SOIL & WEATHER INPUT
AI CROP RECOMMENDATION
TOP 3 CROP RISK ANALYSIS
FINAL DECISION
FARMER ADVICE
```

## Run FastAPI Web App

```bash
uvicorn frontend.app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Pipeline API

Main functions in `pipeline.py`:

```python
load_artifacts()
predict_crop(input_data)
build_risk_features(area, item, year, rainfall, pesticides, avg_temp)
predict_yield_risk(input_data)
make_decision(yield_risk)
predict_full_pipeline(input_data)
run_pipeline(input_data)
```

Sample input:

```python
sample_input = {
    "N": 80,
    "P": 45,
    "K": 20,
    "temperature": 22.5,
    "humidity": 65,
    "ph": 6.3,
    "rainfall": 850,
    "area": "Pakistan",
    "year": 2024,
    "pesticides_tonnes": 12000,
}
```

Expected output keys:

```text
primary_recommended_crop
crop_confidence
top_3_predictions
crop_risk_analysis
best_final_crop
final_risk
final_decision
final_advice
risk_probabilities
llm_advice_used
```

## Notes

- Do not commit `.env`.
- Do not retrain or rename model artifacts unless the pipeline is updated accordingly.
- TensorFlow and scikit-learn may print version warnings during model loading; inference still works if tests pass.
