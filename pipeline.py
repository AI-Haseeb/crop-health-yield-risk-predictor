import os
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

CROP_MODEL_PATH = os.path.join(MODELS_DIR, "crop_recommendation_model.h5")
CROP_SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
CROP_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

RISK_MODEL_PATH = os.path.join(MODELS_DIR, "yield_risk_ann.h5")
RISK_SCALER_PATH = os.path.join(MODELS_DIR, "risk_scaler.pkl")
RISK_ENCODER_PATH = os.path.join(MODELS_DIR, "risk_encoder.pkl")

YIELD_DATA_PATH = os.path.join(DATA_DIR, "yield_df.csv")


# =========================
# LOAD MODELS
# =========================

crop_model = load_model(CROP_MODEL_PATH)
crop_scaler = joblib.load(CROP_SCALER_PATH)
crop_encoder = joblib.load(CROP_ENCODER_PATH)

risk_model = load_model(RISK_MODEL_PATH)
risk_scaler = joblib.load(RISK_SCALER_PATH)
risk_encoder = joblib.load(RISK_ENCODER_PATH)


# =========================
# LOAD YIELD DATA FOR RISK FEATURES
# =========================

yield_df = pd.read_csv(YIELD_DATA_PATH)

yield_df.columns = (
    yield_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Keep original training column names.
# Risk scaler was trained with average_rain_fall_mm_per_year and avg_temp.


# =========================
# HELPER FUNCTIONS
# =========================

def validate_input(data):
    required = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    for col in required:
        if col not in data:
            raise ValueError(f"Missing input field: {col}")

    if not (0 <= float(data["ph"]) <= 14):
        raise ValueError("pH must be between 0 and 14")

    if not (0 <= float(data["humidity"]) <= 100):
        raise ValueError("Humidity must be between 0 and 100")

    if float(data["rainfall"]) < 0:
        raise ValueError("Rainfall cannot be negative")

    return True


def get_crop_prediction(data):
    features = np.array([[
        float(data["N"]),
        float(data["P"]),
        float(data["K"]),
        float(data["temperature"]),
        float(data["humidity"]),
        float(data["ph"]),
        float(data["rainfall"]),
    ]])

    scaled_features = crop_scaler.transform(features)

    probabilities = crop_model.predict(scaled_features, verbose=0)[0]

    top_indices = np.argsort(probabilities)[::-1][:3]
    top_crops = crop_encoder.inverse_transform(top_indices)

    return {
        "recommended_crop": str(top_crops[0]),
        "alternatives": [str(crop) for crop in top_crops[1:]],
        "crop_confidence": round(float(probabilities[top_indices[0]]) * 100, 2),
        "top_3_predictions": [
            {
                "crop": str(crop_encoder.inverse_transform([idx])[0]),
                "confidence": round(float(probabilities[idx]) * 100, 2)
            }
            for idx in top_indices
        ]
    }


def get_crop_reference_values(crop_name):
    crop_name = str(crop_name).lower()

    if "item" in yield_df.columns:
        crop_data = yield_df[yield_df["item"].astype(str).str.lower() == crop_name]
    else:
        crop_data = pd.DataFrame()

    if crop_data.empty:
        crop_data = yield_df

    rainfall_mean = crop_data["rainfall"].mean() if "rainfall" in crop_data.columns else 0
    rainfall_std = crop_data["rainfall"].std() if "rainfall" in crop_data.columns else 0

    temp_mean = crop_data["temperature"].mean() if "temperature" in crop_data.columns else 0
    temp_std = crop_data["temperature"].std() if "temperature" in crop_data.columns else 0

    pesticide_mean = crop_data["pesticides_tonnes"].mean() if "pesticides_tonnes" in crop_data.columns else 0
    pesticide_std = crop_data["pesticides_tonnes"].std() if "pesticides_tonnes" in crop_data.columns else 0

    return {
        "rainfall_mean": 0 if pd.isna(rainfall_mean) else rainfall_mean,
        "rainfall_std": 0 if pd.isna(rainfall_std) else rainfall_std,
        "temp_mean": 0 if pd.isna(temp_mean) else temp_mean,
        "temp_std": 0 if pd.isna(temp_std) else temp_std,
        "pesticide_mean": 0 if pd.isna(pesticide_mean) else pesticide_mean,
        "pesticide_std": 0 if pd.isna(pesticide_std) else pesticide_std,
    }


def create_risk_features(data, recommended_crop):
    rainfall = float(data["rainfall"])
    temperature = float(data["temperature"])
    crop_name = str(recommended_crop).lower()

    # Find crop data from yield dataset
    if "item" in yield_df.columns:
        crop_data = yield_df[yield_df["item"].astype(str).str.lower() == crop_name]
    else:
        crop_data = pd.DataFrame()

    # If exact crop not found, use full dataset averages
    if crop_data.empty:
        crop_data = yield_df

    crop_median_rainfall = crop_data["average_rain_fall_mm_per_year"].median()
    crop_median_temp = crop_data["avg_temp"].median()
    crop_median_pesticides = crop_data["pesticides_tonnes"].median()

    rainfall_volatility = abs(rainfall - crop_median_rainfall)
    temp_volatility = abs(temperature - crop_median_temp)
    pesticide_volatility = 0

    pesticides_tonnes = crop_median_pesticides

    weather_volatility_score = rainfall_volatility + temp_volatility

    pesticides_log = np.log1p(pesticides_tonnes)

    year = 2024
    year_index = year - int(yield_df["year"].min()) if "year" in yield_df.columns else 0

    rainfall_temp_interaction = rainfall * temperature
    rainfall_to_temp_ratio = rainfall / (temperature + 1)
    pesticide_to_rainfall_ratio = pesticides_tonnes / (rainfall + 1)

    crop_rainfall_std = crop_data["average_rain_fall_mm_per_year"].std()
    crop_temp_std = crop_data["avg_temp"].std()
    crop_pesticide_std = crop_data["pesticides_tonnes"].std()

    area_crop_rainfall_median = crop_median_rainfall
    area_crop_temp_median = crop_median_temp

    risk_input = pd.DataFrame([{
        "area": "Pakistan",
        "item": recommended_crop,
        "year": year,

        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides_tonnes,
        "avg_temp": temperature,

        "crop_median_rainfall": crop_median_rainfall,
        "crop_median_temp": crop_median_temp,
        "crop_median_pesticides": crop_median_pesticides,

        "rainfall_volatility": rainfall_volatility,
        "temp_volatility": temp_volatility,
        "pesticide_volatility": pesticide_volatility,
        "weather_volatility_score": weather_volatility_score,

        "pesticides_log": pesticides_log,
        "year_index": year_index,
        "rainfall_temp_interaction": rainfall_temp_interaction,
        "rainfall_to_temp_ratio": rainfall_to_temp_ratio,
        "pesticide_to_rainfall_ratio": pesticide_to_rainfall_ratio,

        "crop_rainfall_std": crop_rainfall_std,
        "crop_temp_std": crop_temp_std,
        "crop_pesticide_std": crop_pesticide_std,

        "area_crop_rainfall_median": area_crop_rainfall_median,
        "area_crop_temp_median": area_crop_temp_median,
    }])

    risk_input = risk_input.fillna(0)

    return risk_input


def get_risk_prediction(data, recommended_crop):
    risk_features = create_risk_features(data, recommended_crop)

    processed_features = risk_scaler.transform(risk_features)

    probabilities = risk_model.predict(processed_features, verbose=0)[0]

    risk_index = int(np.argmax(probabilities))
    risk_label = risk_encoder.inverse_transform([risk_index])[0]

    return {
        "yield_risk": str(risk_label).upper(),
        "risk_confidence": round(float(probabilities[risk_index]) * 100, 2),
        "risk_probabilities": {
            str(risk_encoder.inverse_transform([i])[0]).upper(): round(float(probabilities[i]) * 100, 2)
            for i in range(len(probabilities))
        }
    }


def get_decision(risk):
    risk = risk.upper()

    if risk == "LOW":
        return "GO"
    elif risk == "MEDIUM":
        return "CAUTION"
    else:
        return "HOLD"


def generate_farmer_advice(crop, risk, decision, alternatives):
    if decision == "GO":
        return (
            f"{crop} is suitable for the given soil and weather conditions. "
            f"The yield risk is low, so the farmer can proceed with this crop."
        )

    if decision == "CAUTION":
        return (
            f"{crop} is suitable, but the yield risk is medium. "
            f"The farmer should monitor rainfall and keep irrigation backup ready."
        )

    return (
        f"{crop} has high yield risk under the current conditions. "
        f"The farmer should consider alternative crops such as {alternatives[0]} or {alternatives[1]}."
    )


# =========================
# FINAL PIPELINE
# =========================

def run_pipeline(data):
    validate_input(data)

    crop_result = get_crop_prediction(data)

    risk_result = get_risk_prediction(
        data,
        crop_result["recommended_crop"]
    )

    decision = get_decision(risk_result["yield_risk"])

    advice = generate_farmer_advice(
        crop_result["recommended_crop"],
        risk_result["yield_risk"],
        decision,
        crop_result["alternatives"]
    )

    return {
        "recommended_crop": crop_result["recommended_crop"],
        "alternatives": crop_result["alternatives"],
        "crop_confidence": crop_result["crop_confidence"],
        "top_3_predictions": crop_result["top_3_predictions"],
        "yield_risk": risk_result["yield_risk"],
        "risk_confidence": risk_result["risk_confidence"],
        "risk_probabilities": risk_result["risk_probabilities"],
        "decision": decision,
        "advice": advice
    }


# =========================
# TEST
# =========================

if __name__ == "__main__":
    sample_input = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.0
    }

    result = run_pipeline(sample_input)

    print("\n===== FINAL FARMER OUTPUT =====")
    print(f"Recommended Crop : {result['recommended_crop']}")
    print(f"Alternatives     : {result['alternatives']}")
    print(f"Crop Confidence  : {result['crop_confidence']}%")
    print(f"Yield Risk       : {result['yield_risk']}")
    print(f"Risk Confidence  : {result['risk_confidence']}%")
    print(f"Decision         : {result['decision']}")
    print(f"Advice           : {result['advice']}")