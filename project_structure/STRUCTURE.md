# Crop Health & Yield Risk Predictor Structure

This is the clean working structure for the project.

```text
crop-health-yield-risk-predictor/
  frontend/
    app.py                         FastAPI web app
    templates/
      index.html                   Input form page
      result.html                  Prediction dashboard page
    static/
      css/
        style.css                  Frontend styling

  models/
    crop_recommendation_model.h5   Crop recommendation ANN
    scaler.pkl                     Crop input scaler
    label_encoder.pkl              Crop label encoder
    yield_risk_ann.h5              Yield risk ANN
    risk_scaler.pkl                Risk feature scaler/encoder pipeline
    risk_encoder.pkl               Risk label encoder
    risk_model/
      crop_reference_stats.csv
      area_crop_reference_stats.csv
      yield_risk_features.json

  data/
    raw/
      yield_df.csv                 Yield/risk reference data
      Crop_recommendation.csv      Crop training/reference data
      rainfall.csv
      pesticides.csv
      temp.csv
      yield.csv

  pipeline.py                      Complete ML inference pipeline
  test_pipeline.py                 Terminal-only pipeline test/report
  requirements.txt                 Python dependencies
  README.md                        Project overview
```

Run the FastAPI app:

```bash
uvicorn frontend.app:app --reload
```

Run the terminal pipeline report:

```bash
python test_pipeline.py
```
