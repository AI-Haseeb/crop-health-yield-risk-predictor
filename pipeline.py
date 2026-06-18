# Final pipeline: chains crop recommendation model and yield risk model.
import numpy as np
import joblib
from tensorflow.keras.models import load_model


CROP_MODEL_PATH = "models/crop_recommendation_ann.h5"
RISK_MODEL_PATH = "models/yield_risk_ann.h5"

CROP_SCALER_PATH = "models/crop_scaler.pkl"
RISK_SCALER_PATH = "models/risk_scaler.pkl"

CROP_ENCODER_PATH = "models/crop_encoder.pkl"
RISK_ENCODER_PATH = "models/risk_encoder.pkl"


crop_model = load_model(CROP_MODEL_PATH)
risk_model = load_model(RISK_MODEL_PATH)

crop_scaler = joblib.load(CROP_SCALER_PATH)
risk_scaler = joblib.load(RISK_SCALER_PATH)

crop_encoder = joblib.load(CROP_ENCODER_PATH)
risk_encoder = joblib.load(RISK_ENCODER_PATH)


def nutrient_balance_score(n, p, k):
    return (n + p + k) / 3


def get_decision(risk):
    risk = risk.upper()

    if risk == "LOW":
        return "GO"
    elif risk == "MEDIUM":
        return "CAUTION"
    else:
        return "HOLD"


def generate_advice(crop, risk, decision, alternatives):
    if decision == "GO":
        return f"{crop} is suitable for the given soil and weather conditions. Yield risk is low, so the farmer can proceed."

    elif decision == "CAUTION":
        return f"{crop} is suitable, but yield risk is medium. Farmer should monitor rainfall and keep irrigation backup."

    else:
        return f"{crop} has high yield risk. Farmer should consider alternative crops such as {alternatives[0]} or {alternatives[1]}."


def predict_crop(input_data):
    n = input_data["N"]
    p = input_data["P"]
    k = input_data["K"]
    temperature = input_data["temperature"]
    humidity = input_data["humidity"]
    ph = input_data["ph"]
    rainfall = input_data["rainfall"]

    nutrient_score = nutrient_balance_score(n, p, k)

    features = np.array([[
        n, p, k, temperature, humidity, ph, rainfall, nutrient_score
    ]])

    scaled_features = crop_scaler.transform(features)

    probabilities = crop_model.predict(scaled_features)[0]

    top_indices = np.argsort(probabilities)[::-1][:3]

    top_crops = crop_encoder.inverse_transform(top_indices)

    return {
        "recommended_crop": top_crops[0],
        "alternatives": list(top_crops[1:]),
        "crop_confidence": float(probabilities[top_indices[0]])
    }


def predict_risk(input_data, recommended_crop):
    rainfall = input_data["rainfall"]
    temperature = input_data["temperature"]

    crop_encoded = 0

    rainfall_volatility = 0
    temp_range = 0
    pesticides_tonnes = 0
    yield_trend = 0

    risk_features = np.array([[
        crop_encoded,
        rainfall,
        temperature,
        pesticides_tonnes,
        rainfall_volatility,
        temp_range,
        yield_trend
    ]])

    scaled_risk_features = risk_scaler.transform(risk_features)

    risk_probabilities = risk_model.predict(scaled_risk_features)[0]

    risk_index = np.argmax(risk_probabilities)
    risk = risk_encoder.inverse_transform([risk_index])[0]

    return {
        "yield_risk": risk,
        "risk_confidence": float(risk_probabilities[risk_index])
    }


def run_pipeline(input_data):
    crop_result = predict_crop(input_data)

    risk_result = predict_risk(
        input_data,
        crop_result["recommended_crop"]
    )

    decision = get_decision(risk_result["yield_risk"])

    advice = generate_advice(
        crop_result["recommended_crop"],
        risk_result["yield_risk"],
        decision,
        crop_result["alternatives"]
    )

    return {
        "recommended_crop": crop_result["recommended_crop"],
        "alternatives": crop_result["alternatives"],
        "crop_confidence": crop_result["crop_confidence"],
        "yield_risk": risk_result["yield_risk"],
        "risk_confidence": risk_result["risk_confidence"],
        "decision": decision,
        "advice": advice
    }


if __name__ == "__main__":
    sample_input = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82,
        "ph": 6.5,
        "rainfall": 202
    }

    result = run_pipeline(sample_input)

    print(result)