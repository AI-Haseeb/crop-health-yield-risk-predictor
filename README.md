# Crop Health & Yield Risk Predictor

A full-stack AgriTech hackathon project that combines two trained ANN models with a FastAPI dashboard. The system recommends a suitable crop, predicts yield risk, converts the risk into a simple farmer decision, and shows clear advice with an optional Groq LLM enhancement.

## Key Features

- Crop Recommendation ANN using soil and weather inputs
- Yield Risk ANN using crop, area, year, rainfall, temperature, and pesticide data
- Clean ML pipeline in `pipeline.py`
- FastAPI web app with Jinja templates
- Modern responsive AgriTech dashboard
- Full-screen agriculture video background
- Live Weather Assist using city/country lookup
- Auto-fill for temperature, humidity, rainfall suggestion, and current year
- Country-wise city comparison using live weather
- City Suitability Map with risk-colored circles
- Built-in SVG map fallback if Leaflet/OpenStreetMap is unavailable
- Crop Choice Radar for top crop alternatives
- AI Farm Intelligence with suitability score and risk meter
- Explainable risk reason cards from engineered Model 2 features
- Optional Groq LLM farmer advice
- Terminal-style report through `test_pipeline.py`

## Model Flow

1. User enters farm inputs:
   - `N`, `P`, `K`
   - `temperature`, `humidity`, `ph`, `rainfall`
   - `area`, `year`, `pesticides_tonnes`

2. Model 1 predicts the best crop:
   - output: `recommended_crop`

3. The predicted crop is passed into Model 2:
   - Model 1 crop output becomes Model 2 `item`
   - `rainfall` maps to `average_rain_fall_mm_per_year`
   - `temperature` maps to `avg_temp`
   - `area`, `year`, and `pesticides_tonnes` come from the form

4. Model 2 predicts yield risk:
   - `LOW`
   - `MEDIUM`
   - `HIGH`

5. Decision engine converts risk into farmer action:

```text
LOW    -> GO
MEDIUM -> CAUTION
HIGH   -> HOLD
```

## Shared Weather Input Mapping

The frontend asks for rainfall and temperature only once. The same values are reused across both models:

```text
Crop model rainfall      -> crop input rainfall
Crop model temperature   -> crop input temperature
Risk model rainfall      -> average_rain_fall_mm_per_year
Risk model temperature   -> avg_temp
```

This keeps the form simple and avoids asking farmers to enter duplicate weather values.

## Model 2 Feature Engineering

Model 2 uses the base inputs plus engineered features that compare current conditions with historical crop reference values:

```text
crop_median_rainfall
crop_median_temp
crop_median_pesticides
rainfall_volatility
temp_volatility
pesticide_volatility
weather_volatility_score
pesticides_log
year_index
rainfall_temp_interaction
rainfall_to_temp_ratio
pesticide_to_rainfall_ratio
crop_rainfall_std
crop_temp_std
crop_pesticide_std
area_crop_rainfall_median
area_crop_temp_median
```

These features help the risk model understand whether the current field condition is close to or far from historical normal conditions for that crop.

## Live Weather Assist

The input page includes a Live Weather Assist panel. The user can enter a city and country, then click `Fetch Weather`. The backend uses Open-Meteo geocoding and weather data to auto-fill:

```text
temperature
humidity
rainfall suggestion
year
```

Rainfall remains editable because live weather APIs usually return current or daily rainfall, while the trained models may expect rainfall-scale values from the dataset.

## Country-Wise Location Intelligence

The result page compares the same soil and pesticide inputs across major cities of the selected country. Each city gets live weather, then the pipeline runs again for that city. The dashboard shows:

```text
city
recommended crop
yield risk
final decision
fit score
temperature
humidity
rainfall
```

The City Suitability Map shows the compared cities as colored circles:

```text
GO      -> green circle
CAUTION -> orange circle
HOLD    -> red circle
```

The map uses Leaflet/OpenStreetMap when available. If the map CDN or internet is unavailable, the frontend shows a built-in SVG fallback so the demo still works.

Supported country examples include Pakistan, India, Bangladesh, China, United States/USA, Canada, Mexico, Brazil, Argentina, Australia, Russia, Ukraine, France, Germany, Italy, Spain, United Kingdom/UK, Turkey, Iran, Egypt, Saudi Arabia/KSA, UAE, Nigeria, Ethiopia, Kenya, South Africa, Morocco, Indonesia, Vietnam, Thailand, Philippines, Malaysia, Japan, and South Korea. Unsupported countries fall back to a single-location comparison.

## Area and Year Inputs

`area` gives the yield-risk model regional context. Yield risk can change by country or region because rainfall behavior, pesticide usage, climate patterns, farming practices, and historical yield trends are different across locations.

`year` gives the yield-risk model time context. For real farmer prediction, enter the target/current farming year, for example `2026` if the crop decision is being made for 2026. Past years are useful for historical testing or validation.

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
      videos/
        farm-background.mp4

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

  notebooks/
  reports/
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

Groq is used only to improve farmer advice text. It does not change model predictions, risk labels, or decisions.

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

`.env` is ignored by Git and should not be pushed.

If Groq fails or the key is missing, the app uses built-in fallback advice.

## Run Terminal Pipeline Test

```bash
python test_pipeline.py
```

This prints a clean VS Code terminal report with crop recommendation, top predictions, yield risk, final decision, and farmer advice.

## Run FastAPI Web App

```bash
uvicorn frontend.app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Web Routes

```text
GET  /            -> input dashboard
POST /predict     -> full prediction result page
GET  /weather     -> live city/country weather lookup
POST /api/predict -> JSON prediction used by city cards and map
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

- Do not commit `.env` because it contains the Groq API key.
- Do not retrain or rename model artifacts unless the pipeline is updated accordingly.
- The video background is stored at `frontend/static/videos/farm-background.mp4`.
- If GitHub rejects the video because of size, compress the video before pushing.
- TensorFlow and scikit-learn may print version warnings during model loading; inference still works if tests pass.