"""ML inference pipeline for the Crop Health & Yield Risk Predictor.

This module does not retrain models, rename artifacts, or change model
architecture. It only loads the existing saved artifacts and prepares inference
features in the same order expected by the trained scalers/models.
"""

from __future__ import annotations

import os
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict
from textwrap import fill

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent

CROP_MODEL_PATH = BASE_DIR / "models" / "crop_recommendation_model.h5"
CROP_SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
CROP_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

RISK_MODEL_PATH = BASE_DIR / "models" / "yield_risk_ann.h5"
RISK_SCALER_PATH = BASE_DIR / "models" / "risk_scaler.pkl"
RISK_ENCODER_PATH = BASE_DIR / "models" / "risk_encoder.pkl"

YIELD_DATA_PATH = BASE_DIR / "data" / "raw" / "yield_df.csv"
CROP_REFERENCE_PATH = BASE_DIR / "models" / "risk_model" / "crop_reference_stats.csv"
AREA_CROP_REFERENCE_PATH = BASE_DIR / "models" / "risk_model" / "area_crop_reference_stats.csv"
ENV_PATH = BASE_DIR / ".env"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

RISK_FEATURES = [
    "area",
    "item",
    "year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "crop_median_rainfall",
    "crop_median_temp",
    "crop_median_pesticides",
    "rainfall_volatility",
    "temp_volatility",
    "pesticide_volatility",
    "weather_volatility_score",
    "pesticides_log",
    "year_index",
    "rainfall_temp_interaction",
    "rainfall_to_temp_ratio",
    "pesticide_to_rainfall_ratio",
    "crop_rainfall_std",
    "crop_temp_std",
    "crop_pesticide_std",
    "area_crop_rainfall_median",
    "area_crop_temp_median",
]

LEAKAGE_COLUMNS = {
    "hg_ha_yield",
    "hg/ha_yield",
    "expected_yield",
    "yield_ratio",
    "yield_percentile",
    "yield_risk_score",
    "weather_risk_score",
    "final_risk_score",
    "yield_risk",
}

# The crop recommender predicts simple crop names. The risk model was trained
# with FAO-style crop item names, so known overlaps are mapped before Model 2.
CROP_TO_RISK_ITEM = {
    "rice": "Rice, paddy",
    "maize": "Maize",
    "potato": "Potatoes",
    "soybean": "Soybeans",
    "cassava": "Cassava",
    "yam": "Yams",
}

_ARTIFACTS: Dict[str, Any] | None = None


def _load_env_file() -> None:
    """Load simple KEY=VALUE pairs from .env without printing secrets."""
    if not ENV_PATH.exists() or not ENV_PATH.is_file():
        return

    for line in ENV_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        clean_line = line.strip()
        if not clean_line or clean_line.startswith("#") or "=" not in clean_line:
            continue
        key, value = clean_line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _check_file(path: Path) -> None:
    if not path.exists():
        print(f"Missing required file: {path}")
        raise FileNotFoundError(f"Missing required file: {path}")


def _normalize_yield_data(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    clean_df.columns = (
        clean_df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    clean_df = clean_df.rename(columns={"hg/ha_yield": "hg_ha_yield"})
    return clean_df


def _build_reference_tables(yield_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {
        "area",
        "item",
        "year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    }
    missing_columns = sorted(required_columns - set(yield_df.columns))
    if missing_columns:
        raise ValueError(f"yield_df.csv is missing required columns: {missing_columns}")

    crop_reference = (
        yield_df.groupby("item", as_index=False)
        .agg(
            crop_median_rainfall=("average_rain_fall_mm_per_year", "median"),
            crop_median_temp=("avg_temp", "median"),
            crop_median_pesticides=("pesticides_tonnes", "median"),
            crop_rainfall_std=("average_rain_fall_mm_per_year", "std"),
            crop_temp_std=("avg_temp", "std"),
            crop_pesticide_std=("pesticides_tonnes", "std"),
        )
        .fillna(0)
    )

    area_crop_reference = (
        yield_df.groupby(["area", "item"], as_index=False)
        .agg(
            area_crop_rainfall_median=("average_rain_fall_mm_per_year", "median"),
            area_crop_temp_median=("avg_temp", "median"),
        )
        .fillna(0)
    )

    return crop_reference, area_crop_reference


def _load_reference_tables(yield_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use saved reference CSVs when present, otherwise compute them from yield_df."""
    if CROP_REFERENCE_PATH.exists() and AREA_CROP_REFERENCE_PATH.exists():
        crop_reference = _normalize_yield_data(pd.read_csv(CROP_REFERENCE_PATH))
        area_crop_reference = _normalize_yield_data(pd.read_csv(AREA_CROP_REFERENCE_PATH))

        required_crop_columns = {
            "item",
            "crop_median_rainfall",
            "crop_median_temp",
            "crop_median_pesticides",
            "crop_rainfall_std",
            "crop_temp_std",
            "crop_pesticide_std",
        }
        required_area_crop_columns = {
            "area",
            "item",
            "area_crop_rainfall_median",
            "area_crop_temp_median",
        }
        missing_crop_columns = sorted(required_crop_columns - set(crop_reference.columns))
        missing_area_crop_columns = sorted(
            required_area_crop_columns - set(area_crop_reference.columns)
        )

        if not missing_crop_columns and not missing_area_crop_columns:
            return crop_reference, area_crop_reference

        print("Reference CSV files exist but are missing required columns.")
        if missing_crop_columns:
            print(f"Missing crop reference columns: {missing_crop_columns}")
        if missing_area_crop_columns:
            print(f"Missing area crop reference columns: {missing_area_crop_columns}")
        print("Recreating reference values from data/raw/yield_df.csv.")

    return _build_reference_tables(yield_df)


def load_artifacts() -> Dict[str, Any]:
    """Load models, scalers, encoders, and yield reference data once."""
    global _ARTIFACTS
    if _ARTIFACTS is not None:
        return _ARTIFACTS

    required_files = [
        CROP_MODEL_PATH,
        CROP_SCALER_PATH,
        CROP_ENCODER_PATH,
        RISK_MODEL_PATH,
        RISK_SCALER_PATH,
        RISK_ENCODER_PATH,
        YIELD_DATA_PATH,
    ]
    for path in required_files:
        _check_file(path)

    crop_model = load_model(CROP_MODEL_PATH, compile=False)
    crop_scaler = joblib.load(CROP_SCALER_PATH)
    crop_encoder = joblib.load(CROP_ENCODER_PATH)

    risk_model = load_model(RISK_MODEL_PATH, compile=False)
    risk_scaler = joblib.load(RISK_SCALER_PATH)
    risk_encoder = joblib.load(RISK_ENCODER_PATH)

    yield_df = _normalize_yield_data(pd.read_csv(YIELD_DATA_PATH))
    crop_reference, area_crop_reference = _load_reference_tables(yield_df)
    min_year = int(yield_df["year"].min())

    scaler_features = list(getattr(risk_scaler, "feature_names_in_", []))
    if scaler_features and scaler_features != RISK_FEATURES:
        raise ValueError("risk_scaler.pkl feature order does not match pipeline RISK_FEATURES.")

    _ARTIFACTS = {
        "crop_model": crop_model,
        "crop_scaler": crop_scaler,
        "crop_encoder": crop_encoder,
        "risk_model": risk_model,
        "risk_scaler": risk_scaler,
        "risk_encoder": risk_encoder,
        "yield_df": yield_df,
        "crop_reference": crop_reference,
        "area_crop_reference": area_crop_reference,
        "min_year": min_year,
    }
    return _ARTIFACTS


def _require_fields(input_data: Dict[str, Any], fields: list[str]) -> None:
    for field in fields:
        if field not in input_data or input_data[field] in ("", None):
            raise ValueError(f"Missing input field: {field}")


def _to_float(input_data: Dict[str, Any], field: str) -> float:
    try:
        return float(input_data[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc


def _to_int(input_data: Dict[str, Any], field: str) -> int:
    try:
        return int(input_data[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer.") from exc


def _validate_full_input(input_data: Dict[str, Any]) -> None:
    _require_fields(
        input_data,
        [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
            "area",
            "year",
            "pesticides_tonnes",
        ],
    )

    ph = _to_float(input_data, "ph")
    humidity = _to_float(input_data, "humidity")
    rainfall = _to_float(input_data, "rainfall")

    if not 0 <= ph <= 14:
        raise ValueError("ph must be between 0 and 14.")
    if not 0 <= humidity <= 100:
        raise ValueError("humidity must be between 0 and 100.")
    if rainfall < 0:
        raise ValueError("rainfall cannot be negative.")


def _format_probability_label(value: float) -> str:
    """Format tiny probabilities without hiding them as plain 0.0%."""
    value = float(value)
    if 0 < value < 0.01:
        return "<0.01"
    if value >= 99.995:
        return "100.00"
    return f"{value:.2f}"


def _probabilities_to_dict(classes: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    output = {}
    for label, probability in zip(classes, probabilities):
        label_text = str(label)
        if label_text.upper() in {"LOW", "MEDIUM", "HIGH"}:
            label_text = label_text.upper()
        # Keep enough precision so top predictions are not lost by early rounding.
        output[label_text] = round(float(probability) * 100, 6)
    return output


def _probability_items(probabilities: Dict[str, float]) -> list[Dict[str, Any]]:
    return [
        {
            "label": label,
            "probability": probability,
            "probability_display": _format_probability_label(probability),
        }
        for label, probability in probabilities.items()
    ]


def _top_crop_predictions(classes: np.ndarray, probabilities: np.ndarray, limit: int = 3) -> list[Dict[str, Any]]:
    pairs = sorted(
        zip(classes, probabilities),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:limit]
    return [
        {
            "crop": str(crop),
            "confidence": round(float(probability) * 100, 6),
            "confidence_display": _format_probability_label(float(probability) * 100),
        }
        for crop, probability in pairs
    ]


def _model2_item_name(crop_name: str) -> str:
    crop_text = str(crop_name).strip()
    return CROP_TO_RISK_ITEM.get(crop_text.lower(), crop_text)


def predict_crop(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run Model 1 and return the recommended crop plus class probabilities."""
    artifacts = load_artifacts()
    _require_fields(input_data, CROP_FEATURES)

    crop_values = np.array([[_to_float(input_data, field) for field in CROP_FEATURES]])
    crop_values_scaled = artifacts["crop_scaler"].transform(crop_values)
    probabilities = artifacts["crop_model"].predict(crop_values_scaled, verbose=0)[0]

    best_index = int(np.argmax(probabilities))
    recommended_crop = str(artifacts["crop_encoder"].inverse_transform([best_index])[0])

    return {
        "recommended_crop": recommended_crop,
        "crop_probabilities": _probabilities_to_dict(
            artifacts["crop_encoder"].classes_,
            probabilities,
        ),
        "top_3_predictions": _top_crop_predictions(
            artifacts["crop_encoder"].classes_,
            probabilities,
        ),
    }


def _crop_stats(item: str, crop_reference: pd.DataFrame) -> Dict[str, float]:
    matched = crop_reference[crop_reference["item"].astype(str).str.lower() == item.lower()]

    if matched.empty:
        numeric_reference = crop_reference.select_dtypes(include=[np.number])
        return {
            "crop_median_rainfall": float(numeric_reference["crop_median_rainfall"].median()),
            "crop_median_temp": float(numeric_reference["crop_median_temp"].median()),
            "crop_median_pesticides": float(numeric_reference["crop_median_pesticides"].median()),
            "crop_rainfall_std": float(numeric_reference["crop_rainfall_std"].median()),
            "crop_temp_std": float(numeric_reference["crop_temp_std"].median()),
            "crop_pesticide_std": float(numeric_reference["crop_pesticide_std"].median()),
        }

    row = matched.iloc[0]
    return {
        "crop_median_rainfall": float(row["crop_median_rainfall"]),
        "crop_median_temp": float(row["crop_median_temp"]),
        "crop_median_pesticides": float(row["crop_median_pesticides"]),
        "crop_rainfall_std": float(row["crop_rainfall_std"]),
        "crop_temp_std": float(row["crop_temp_std"]),
        "crop_pesticide_std": float(row["crop_pesticide_std"]),
    }


def _area_crop_stats(
    area: str,
    item: str,
    area_crop_reference: pd.DataFrame,
    fallback_crop_stats: Dict[str, float],
) -> Dict[str, float]:
    matched = area_crop_reference[
        (area_crop_reference["area"].astype(str).str.lower() == area.lower())
        & (area_crop_reference["item"].astype(str).str.lower() == item.lower())
    ]

    if matched.empty:
        return {
            "area_crop_rainfall_median": fallback_crop_stats["crop_median_rainfall"],
            "area_crop_temp_median": fallback_crop_stats["crop_median_temp"],
        }

    row = matched.iloc[0]
    return {
        "area_crop_rainfall_median": float(row["area_crop_rainfall_median"]),
        "area_crop_temp_median": float(row["area_crop_temp_median"]),
    }


def build_risk_features(
    area: str,
    item: str,
    year: int,
    rainfall: float,
    pesticides: float,
    avg_temp: float,
) -> pd.DataFrame:
    """Build Model 2 features without leakage columns."""
    artifacts = load_artifacts()

    area_value = str(area).strip()
    item_value = _model2_item_name(item)
    year_value = int(year)
    rainfall_value = float(rainfall)
    pesticides_value = float(pesticides)
    avg_temp_value = float(avg_temp)
    eps = 1e-6

    crop_reference_values = _crop_stats(item_value, artifacts["crop_reference"])
    area_crop_reference_values = _area_crop_stats(
        area_value,
        item_value,
        artifacts["area_crop_reference"],
        crop_reference_values,
    )

    crop_median_rainfall = crop_reference_values["crop_median_rainfall"]
    crop_median_temp = crop_reference_values["crop_median_temp"]
    crop_median_pesticides = crop_reference_values["crop_median_pesticides"]

    rainfall_volatility = abs(rainfall_value - crop_median_rainfall) / (
        abs(crop_median_rainfall) + eps
    )
    temp_volatility = abs(avg_temp_value - crop_median_temp) / (abs(crop_median_temp) + eps)
    pesticide_volatility = abs(pesticides_value - crop_median_pesticides) / (
        abs(crop_median_pesticides) + eps
    )

    feature_values = {
        "area": area_value,
        "item": item_value,
        "year": year_value,
        "average_rain_fall_mm_per_year": rainfall_value,
        "pesticides_tonnes": pesticides_value,
        "avg_temp": avg_temp_value,
        "crop_median_rainfall": crop_median_rainfall,
        "crop_median_temp": crop_median_temp,
        "crop_median_pesticides": crop_median_pesticides,
        "rainfall_volatility": rainfall_volatility,
        "temp_volatility": temp_volatility,
        "pesticide_volatility": pesticide_volatility,
        "weather_volatility_score": (
            rainfall_volatility * 0.45
            + temp_volatility * 0.45
            + pesticide_volatility * 0.10
        ),
        "pesticides_log": float(np.log1p(pesticides_value)),
        "year_index": year_value - artifacts["min_year"],
        "rainfall_temp_interaction": rainfall_value * avg_temp_value,
        "rainfall_to_temp_ratio": rainfall_value / (avg_temp_value + eps),
        "pesticide_to_rainfall_ratio": pesticides_value / (rainfall_value + eps),
        **crop_reference_values,
        **area_crop_reference_values,
    }

    risk_features = pd.DataFrame([feature_values], columns=RISK_FEATURES).fillna(0)
    leakage_columns_used = sorted(set(risk_features.columns) & LEAKAGE_COLUMNS)
    if leakage_columns_used:
        raise ValueError(f"Leakage columns are not allowed: {leakage_columns_used}")

    return risk_features


def predict_yield_risk(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run Model 2 and return LOW, MEDIUM, or HIGH risk plus probabilities."""
    artifacts = load_artifacts()
    _require_fields(
        input_data,
        [
            "area",
            "item",
            "year",
            "average_rain_fall_mm_per_year",
            "pesticides_tonnes",
            "avg_temp",
        ],
    )

    risk_features = build_risk_features(
        area=input_data["area"],
        item=input_data["item"],
        year=_to_int(input_data, "year"),
        rainfall=_to_float(input_data, "average_rain_fall_mm_per_year"),
        pesticides=_to_float(input_data, "pesticides_tonnes"),
        avg_temp=_to_float(input_data, "avg_temp"),
    )

    risk_values_scaled = artifacts["risk_scaler"].transform(risk_features)
    probabilities = artifacts["risk_model"].predict(risk_values_scaled, verbose=0)[0]

    best_index = int(np.argmax(probabilities))
    yield_risk = str(artifacts["risk_encoder"].inverse_transform([best_index])[0]).upper()

    return {
        "yield_risk": yield_risk,
        "risk_probabilities": _probabilities_to_dict(
            artifacts["risk_encoder"].classes_,
            probabilities,
        ),
    }


def make_decision(yield_risk: str) -> str:
    """Convert Model 2 risk label into the farmer decision."""
    risk = str(yield_risk).upper()
    if risk == "LOW":
        return "GO"
    if risk == "MEDIUM":
        return "CAUTION"
    if risk == "HIGH":
        return "HOLD"
    raise ValueError(f"Unknown yield risk label: {yield_risk}")


def _make_advice(crop: str, yield_risk: str, decision: str) -> str:
    if decision == "GO":
        return (
            f"{crop} is recommended and the predicted yield risk is LOW. "
            "The farmer can proceed while continuing normal field monitoring."
        )
    if decision == "CAUTION":
        return (
            f"{crop} is recommended, but the predicted yield risk is MEDIUM. "
            "Proceed carefully, monitor weather, and keep mitigation support ready."
        )
    return (
        f"{crop} is recommended by the crop model, but the predicted yield risk is HIGH. "
        "The farmer should hold planting and review local weather, irrigation, and expert advice."
    )


def _make_llm_advice(
    input_data: Dict[str, Any],
    crop: str,
    yield_risk: str,
    decision: str,
    fallback_advice: str,
) -> tuple[str, bool]:
    """Use Groq only to improve farmer advice. Never let it change ML outputs."""
    _load_env_file()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return fallback_advice, False

    prompt = f"""
You are an agricultural advisory assistant.

Write a concise, farmer-friendly advisory for the given ML prediction.
Do not change the crop, yield risk, or final decision.
Do not claim certainty. Mention that this is decision support and local expert/weather checks still matter.
Keep it practical and under 90 words.

Inputs:
- N: {input_data["N"]}
- P: {input_data["P"]}
- K: {input_data["K"]}
- temperature: {input_data["temperature"]}
- humidity: {input_data["humidity"]}
- ph: {input_data["ph"]}
- rainfall: {input_data["rainfall"]}
- area: {input_data["area"]}
- year: {input_data["year"]}
- pesticides_tonnes: {input_data["pesticides_tonnes"]}

ML prediction:
- recommended_crop: {crop}
- yield_risk: {yield_risk}
- final_decision: {decision}
""".strip()

    payload = {
        "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You produce practical agricultural advice from fixed ML outputs. "
                    "Never alter the provided crop, risk, or decision."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 180,
    }

    request = urllib.request.Request(
        GROQ_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 CropHealthYieldRiskPredictor/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            response_data = json.loads(response.read().decode("utf-8"))
        llm_advice = response_data["choices"][0]["message"]["content"].strip()
        return llm_advice or fallback_advice, bool(llm_advice)
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError, TimeoutError):
        return fallback_advice, False


def predict_full_pipeline(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run Model 1, feed its crop into Model 2, and return the final result."""
    _validate_full_input(input_data)

    crop_result = predict_crop(input_data)
    recommended_crop = crop_result["recommended_crop"]

    risk_input = {
        "area": input_data["area"],
        "item": recommended_crop,
        "year": input_data["year"],
        # Keep the form simple: one rainfall/temperature entry feeds both models.
        "average_rain_fall_mm_per_year": input_data["rainfall"],
        "pesticides_tonnes": input_data["pesticides_tonnes"],
        "avg_temp": input_data["temperature"],
    }

    risk_result = predict_yield_risk(risk_input)
    yield_risk = risk_result["yield_risk"]
    decision = make_decision(yield_risk)
    crop_probabilities = crop_result["crop_probabilities"]
    risk_probabilities = risk_result["risk_probabilities"]
    top_3_predictions = crop_result["top_3_predictions"]
    crop_confidence = top_3_predictions[0]["confidence"] if top_3_predictions else 0.0
    crop_confidence_display = (
        top_3_predictions[0]["confidence_display"] if top_3_predictions else "0.00"
    )
    risk_probability_items = _probability_items(risk_probabilities)
    crop_risk_analysis = []
    for crop_item in top_3_predictions:
        analysis_risk_input = {
            "area": input_data["area"],
            "item": crop_item["crop"],
            "year": input_data["year"],
            "average_rain_fall_mm_per_year": input_data["rainfall"],
            "pesticides_tonnes": input_data["pesticides_tonnes"],
            "avg_temp": input_data["temperature"],
        }
        analysis_risk_result = predict_yield_risk(analysis_risk_input)
        analysis_risk = analysis_risk_result["yield_risk"]
        analysis_decision = make_decision(analysis_risk)
        analysis_probabilities = analysis_risk_result["risk_probabilities"]
        crop_risk_analysis.append(
            {
                "crop": crop_item["crop"],
                "yield_risk": analysis_risk,
                "risk_confidence": max(analysis_probabilities.values()),
                "risk_confidence_display": _format_probability_label(
                    max(analysis_probabilities.values())
                ),
                "decision": analysis_decision,
                "risk_probabilities": analysis_probabilities,
            }
        )

    fallback_advice = _make_advice(recommended_crop, yield_risk, decision)
    advice, llm_advice_used = _make_llm_advice(
        input_data=input_data,
        crop=recommended_crop,
        yield_risk=yield_risk,
        decision=decision,
        fallback_advice=fallback_advice,
    )

    return {
        "recommended_crop": recommended_crop,
        "yield_risk": yield_risk,
        "decision": decision,
        "advice": advice,
        "crop_probabilities": crop_probabilities,
        "risk_probabilities": risk_probabilities,
        "risk_probability_items": risk_probability_items,
        "primary_recommended_crop": recommended_crop,
        "crop_confidence": crop_confidence,
        "crop_confidence_display": crop_confidence_display,
        "top_3_predictions": top_3_predictions,
        "crop_risk_analysis": crop_risk_analysis,
        "best_final_crop": recommended_crop,
        "final_risk": yield_risk,
        "final_decision": decision,
        "final_advice": advice,
        "llm_advice_used": llm_advice_used,
    }


def run_pipeline(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible Flask entry point."""
    return predict_full_pipeline(input_data)


def _percent(value: Any) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def print_console_report(input_data: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Print a clean terminal report for local CLI testing."""
    line = "=" * 55
    subline = "-" * 55

    print("\n" + line)
    print("        CROP HEALTH & YIELD RISK PREDICTION REPORT")
    print(line)

    print("\nSOIL & WEATHER INPUT")
    print(subline)
    for key in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
        print(f"{key}: {input_data[key]}")

    print("\nAI CROP RECOMMENDATION")
    print(subline)
    print(f"Primary Recommended Crop : {result['primary_recommended_crop']}")
    print(f"Crop Confidence          : {_percent(result.get('crop_confidence'))}")

    print("\nTOP 3 CROP RISK ANALYSIS")
    print(subline)
    for item in result.get("crop_risk_analysis", []):
        print(
            f"{item.get('crop', 'N/A')} | "
            f"Risk: {item.get('yield_risk', 'N/A')} | "
            f"Confidence: {_percent(item.get('risk_confidence'))} | "
            f"Decision: {item.get('decision', 'N/A')}"
        )

    print("\nFINAL DECISION")
    print(subline)
    print(f"Best Final Crop : {result['best_final_crop']}")
    print(f"Final Risk      : {result['final_risk']}")
    print(f"Decision        : {result['final_decision']}")

    print("\nFARMER ADVICE")
    print(subline)
    print(fill(result["final_advice"], width=110))


if __name__ == "__main__":
    sample_input = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.0,
        "area": "Pakistan",
        "year": 2024,
        "pesticides_tonnes": 12000,
    }
    pipeline_result = run_pipeline(sample_input)
    print_console_report(sample_input, pipeline_result)
