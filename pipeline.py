"""ML inference pipeline for the Crop Health & Yield Risk Predictor.

This module does not retrain models, rename artifacts, or change model
architecture. It only loads the existing saved artifacts and prepares inference
features in the same order expected by the trained scalers/models.
"""

from __future__ import annotations  # Allows modern type hints to work safely.

import os  # Reads environment variables such as GROQ_API_KEY.
import json  # Converts Python dictionaries to/from JSON for Groq API responses.
import urllib.error  # Handles web/API errors from the optional Groq request.
import urllib.request  # Sends the optional Groq API HTTP request.
from pathlib import Path  # Builds file paths for models, data, and .env.
from typing import Any, Dict  # Adds readable type hints for dictionaries.
from textwrap import fill  # Wraps long advice text nicely in terminal output.

import joblib  # Loads saved sklearn scalers and encoders from .pkl files.
import numpy as np  # Handles numeric arrays, argmax, log transforms, and math.
import pandas as pd  # Reads CSV files and builds DataFrames for Model 2 features.
from tensorflow.keras.models import load_model  # Loads saved ANN .h5 model files.


# Project root path. All model/data paths are built from this folder.
BASE_DIR = Path(__file__).resolve().parent

# Saved Model 1 ANN file for crop recommendation.
CROP_MODEL_PATH = BASE_DIR / "models" / "crop_recommendation_model.h5"
# Scaler used during Model 1 training; same scaler must be used during prediction.
CROP_SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
# Label encoder converts Model 1 numeric output back into crop names.
CROP_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

# Saved Model 2 ANN file for yield risk prediction.
RISK_MODEL_PATH = BASE_DIR / "models" / "yield_risk_ann.h5"
# Scaler/transformer used during Model 2 training.
RISK_SCALER_PATH = BASE_DIR / "models" / "risk_scaler.pkl"
# Encoder converts Model 2 numeric output back into LOW/MEDIUM/HIGH.
RISK_ENCODER_PATH = BASE_DIR / "models" / "risk_encoder.pkl"

# Historical yield dataset used only for reference statistics, not retraining.
YIELD_DATA_PATH = BASE_DIR / "data" / "raw" / "yield_df.csv"
CROP_REFERENCE_PATH = BASE_DIR / "models" / "risk_model" / "crop_reference_stats.csv"
AREA_CROP_REFERENCE_PATH = BASE_DIR / "models" / "risk_model" / "area_crop_reference_stats.csv"
ENV_PATH = BASE_DIR / ".env"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Exact feature order expected by the crop recommendation scaler/model.
CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Exact final feature order expected by the yield risk scaler/model.
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

# These columns are target/result columns, so they are blocked from model input.
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

# Cache for loaded models/scalers/encoders so they load once, not on every request.
_ARTIFACTS: Dict[str, Any] | None = None


# Loads optional Groq API key from .env for AI advice.
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


# Gives a clear error if an expected model/data file is missing.
def _check_file(path: Path) -> None:
    if not path.exists():
        print(f"Missing required file: {path}")
        raise FileNotFoundError(f"Missing required file: {path}")


# Standardizes dataset column names so later code can use one naming style.
def _normalize_yield_data(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    clean_df.columns = (
        clean_df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    clean_df = clean_df.rename(columns={"hg/ha_yield": "hg_ha_yield"})
    return clean_df


# Creates crop and area-crop reference values from the historical yield dataset.
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

    # Crop-level historical medians/std values used for volatility comparison.
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

    # Area+crop historical medians give more location-specific reference values.
    # Crop-level historical medians/std values used for volatility comparison.
    area_crop_reference = (
        yield_df.groupby(["area", "item"], as_index=False)
        .agg(
            area_crop_rainfall_median=("average_rain_fall_mm_per_year", "median"),
            area_crop_temp_median=("avg_temp", "median"),
        )
        .fillna(0)
    )

    return crop_reference, area_crop_reference


# Loads saved reference CSVs if present; otherwise recreates them from yield_df.csv.
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


# Central artifact loader used by all prediction functions.
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

    crop_model = load_model(CROP_MODEL_PATH, compile=False)  # Load Model 1 without recompiling training settings.
    crop_scaler = joblib.load(CROP_SCALER_PATH)  # Load Model 1 scaler.
    crop_encoder = joblib.load(CROP_ENCODER_PATH)  # Load crop label encoder.

    risk_model = load_model(RISK_MODEL_PATH, compile=False)  # Load Model 2 without retraining.
    risk_scaler = joblib.load(RISK_SCALER_PATH)  # Load Model 2 preprocessing transformer.
    risk_encoder = joblib.load(RISK_ENCODER_PATH)  # Load risk label encoder.

    yield_df = _normalize_yield_data(pd.read_csv(YIELD_DATA_PATH))  # Read historical yield data for references.
    crop_reference, area_crop_reference = _load_reference_tables(yield_df)
    min_year = int(yield_df["year"].min())  # Baseline year used to create year_index.

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


# Checks that all required inputs are present before prediction.
def _require_fields(input_data: Dict[str, Any], fields: list[str]) -> None:
    for field in fields:
        if field not in input_data or input_data[field] in ("", None):
            raise ValueError(f"Missing input field: {field}")


# Converts form/string values into float numbers and shows clear errors.
def _to_float(input_data: Dict[str, Any], field: str) -> float:
    try:
        return float(input_data[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc


# Converts form/string values into integer numbers and shows clear errors.
def _to_int(input_data: Dict[str, Any], field: str) -> int:
    try:
        return int(input_data[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer.") from exc


# Validates the complete web form input before running both models.
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


# Converts raw model probabilities into readable percentages by class label.
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


# Sorts crop probabilities and keeps the top predictions for dashboard display.
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


# Converts simple crop names into risk-model item names when a mapping exists.
def _model2_item_name(crop_name: str) -> str:
    crop_text = str(crop_name).strip()
    return CROP_TO_RISK_ITEM.get(crop_text.lower(), crop_text)


def predict_crop(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run Model 1 and return the recommended crop plus class probabilities."""
    artifacts = load_artifacts()
    _require_fields(input_data, CROP_FEATURES)

    crop_values = np.array([[_to_float(input_data, field) for field in CROP_FEATURES]])  # Build Model 1 input in exact feature order.
    crop_values_scaled = artifacts["crop_scaler"].transform(crop_values)  # Apply same scaling used during training.
    probabilities = artifacts["crop_model"].predict(crop_values_scaled, verbose=0)[0]  # Get crop class probabilities.

    best_index = int(np.argmax(probabilities))  # Pick the crop class with the highest probability.
    recommended_crop = str(artifacts["crop_encoder"].inverse_transform([best_index])[0])  # Decode class index into crop name.

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


# Finds historical crop reference values; falls back to global medians if crop is unknown.
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


# Finds area+crop reference values; falls back to crop-level values if missing.
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


# Builds all engineered features required by Model 2.
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
    item_value = _model2_item_name(item)  # Map crop name to yield dataset item name when possible.
    year_value = int(year)
    rainfall_value = float(rainfall)
    pesticides_value = float(pesticides)
    avg_temp_value = float(avg_temp)
    eps = 1e-6  # Small value to avoid division by zero in ratio features.

    crop_reference_values = _crop_stats(item_value, artifacts["crop_reference"])  # Get historical crop medians/std values.
    area_crop_reference_values = _area_crop_stats(
        area_value,
        item_value,
        artifacts["area_crop_reference"],
        crop_reference_values,
    )

    crop_median_rainfall = crop_reference_values["crop_median_rainfall"]
    crop_median_temp = crop_reference_values["crop_median_temp"]
    crop_median_pesticides = crop_reference_values["crop_median_pesticides"]

    # Compare current rainfall with historical normal rainfall for that crop.
    rainfall_volatility = abs(rainfall_value - crop_median_rainfall) / (
        abs(crop_median_rainfall) + eps
    )
    temp_volatility = abs(avg_temp_value - crop_median_temp) / (abs(crop_median_temp) + eps)  # Compare current temperature with crop normal.
    # Compare current pesticide usage with historical crop normal.
    pesticide_volatility = abs(pesticides_value - crop_median_pesticides) / (
        abs(crop_median_pesticides) + eps
    )

    # Final dictionary contains base features plus engineered risk features.
    feature_values = {
        "area": area_value,
        "item": item_value,
        "year": year_value,
        "average_rain_fall_mm_per_year": rainfall_value,  # Frontend rainfall is reused for Model 2 rainfall.
        "pesticides_tonnes": pesticides_value,
        "avg_temp": avg_temp_value,  # Frontend temperature is reused for Model 2 avg_temp.
        "crop_median_rainfall": crop_median_rainfall,
        "crop_median_temp": crop_median_temp,
        "crop_median_pesticides": crop_median_pesticides,
        "rainfall_volatility": rainfall_volatility,
        "temp_volatility": temp_volatility,
        "pesticide_volatility": pesticide_volatility,
        "weather_volatility_score": (  # Weighted combined weather/pesticide risk signal.
            rainfall_volatility * 0.45
            + temp_volatility * 0.45
            + pesticide_volatility * 0.10
        ),
        "pesticides_log": float(np.log1p(pesticides_value)),  # Log transform reduces effect of very large pesticide values.
        "year_index": year_value - artifacts["min_year"],  # Converts year into distance from earliest dataset year.
        "rainfall_temp_interaction": rainfall_value * avg_temp_value,  # Captures combined rain-temperature effect.
        "rainfall_to_temp_ratio": rainfall_value / (avg_temp_value + eps),  # Rain per temperature unit.
        "pesticide_to_rainfall_ratio": pesticides_value / (rainfall_value + eps),  # Pesticide amount relative to rainfall.
        **crop_reference_values,
        **area_crop_reference_values,
    }

    risk_features = pd.DataFrame([feature_values], columns=RISK_FEATURES).fillna(0)  # Force exact Model 2 feature order.
    leakage_columns_used = sorted(set(risk_features.columns) & LEAKAGE_COLUMNS)  # Safety check against data leakage.
    if leakage_columns_used:
        raise ValueError(f"Leakage columns are not allowed: {leakage_columns_used}")

    return risk_features


# Runs Model 2 after risk features are built.
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

    risk_features = build_risk_features(  # Convert base risk inputs into the full 23-feature row.
        area=input_data["area"],
        item=input_data["item"],
        year=_to_int(input_data, "year"),
        rainfall=_to_float(input_data, "average_rain_fall_mm_per_year"),
        pesticides=_to_float(input_data, "pesticides_tonnes"),
        avg_temp=_to_float(input_data, "avg_temp"),
    )

    risk_values_scaled = artifacts["risk_scaler"].transform(risk_features)  # Apply same preprocessing used during Model 2 training.
    probabilities = artifacts["risk_model"].predict(risk_values_scaled, verbose=0)[0]  # Get LOW/MEDIUM/HIGH probabilities.

    best_index = int(np.argmax(probabilities))  # Pick the crop class with the highest probability.
    yield_risk = str(artifacts["risk_encoder"].inverse_transform([best_index])[0]).upper()  # Decode risk label.

    return {
        "yield_risk": yield_risk,
        "risk_probabilities": _probabilities_to_dict(
            artifacts["risk_encoder"].classes_,
            probabilities,
        ),
    }


# Simple rule engine that turns risk into farmer-friendly action.
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


# Basic fallback advice used if Groq LLM advice is unavailable.
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


def _strip_markdown_advice(text: str) -> str:
    """Remove simple Markdown markers that look awkward in the dashboard."""
    cleaned = str(text or "").replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.replace("`", "")
    lines = []
    for line in cleaned.splitlines():
        line = line.strip()
        while line.startswith(("- ", "* ")):
            line = line[2:].strip()
        if line:
            lines.append(line)
    return " ".join(lines).strip()


# Structured fallback advice for the dashboard cards.
def _fallback_advice_parts(
    input_data: Dict[str, Any],
    crop: str,
    yield_risk: str,
    decision: str,
) -> Dict[str, Any]:
    crop_title = str(crop).title()
    area = input_data.get("area", "your area")
    year = input_data.get("year", "the target year")
    rainfall = input_data.get("rainfall")
    temperature = input_data.get("temperature")
    humidity = input_data.get("humidity")
    ph = input_data.get("ph")
    pesticides = input_data.get("pesticides_tonnes")

    if decision == "GO":
        summary = f"{crop_title} looks like a good choice for {area} in {year}."
        points = [
            f"The field values look okay for {crop_title}: pH {ph}, temperature {temperature} C, and humidity {humidity}%.",
            f"Use the {rainfall} mm rainfall value to plan water needs and keep checking the crop.",
            "Before planting, check the latest local weather and ask a local agriculture expert if needed.",
        ]
    elif decision == "CAUTION":
        summary = f"{crop_title} may work in {area}, but the risk is medium, so be careful."
        points = [
            f"The model sees some risk for {crop_title} with rainfall {rainfall} mm and temperature {temperature} C.",
            f"Keep extra water support ready and review pesticide use of {pesticides} tonnes.",
            "Before planting, check the weather again and take local expert advice.",
        ]
    else:
        summary = f"{crop_title} is risky in these conditions, so it is better to wait."
        points = [
            f"The risk is high with rainfall {rainfall} mm and temperature {temperature} C.",
            f"Do not spend more on inputs until you review the pesticide plan of {pesticides} tonnes.",
            "Before planting, check other crop options and talk to a local agriculture expert.",
        ]

    return {"summary": summary, "points": points[:3]}

# Converts advice summary/points into one sentence string for console output.
def _format_advice_from_parts(parts: Dict[str, Any]) -> str:
    summary = _strip_markdown_advice(parts.get("summary", ""))
    points = [_strip_markdown_advice(point) for point in parts.get("points", []) if point]
    if points:
        return summary + " " + " ".join(points)
    return summary


# Adds small labels to advice points for the frontend.
def _advice_items(points: list[str]) -> list[Dict[str, str]]:
    labels = ["Field Focus", "Suggested Action", "Before Planting"]
    return [
        {"label": labels[index] if index < len(labels) else "Note", "text": point}
        for index, point in enumerate(points)
    ]

# Optional Groq LLM step: improves wording only, never changes model prediction.
def _make_llm_advice(
    input_data: Dict[str, Any],
    crop: str,
    yield_risk: str,
    decision: str,
    fallback_advice: str,
) -> tuple[Dict[str, Any], bool]:
    """Use Groq only to improve farmer advice. Never let it change ML outputs."""
    fallback_parts = _fallback_advice_parts(input_data, crop, yield_risk, decision)
    _load_env_file()
    api_key = os.getenv("GROQ_API_KEY")  # Reads Groq key from environment/.env if available.
    if not api_key:  # If no key exists, use safe fallback advice.
        return fallback_parts, False

    prompt = f"""
Create a concise farmer advisory from the fixed ML prediction.
Return ONLY valid JSON. Do not use Markdown, asterisks, headings, or bullet symbols.
Do not change the crop, yield risk, or final decision.
Do not claim certainty. Mention that local expert/weather checks still matter.
Use easy English only. Use short, simple words that a farmer can understand. Make every sentence specific to this crop, risk level, area, rainfall, temperature, pH, and pesticide input.
Avoid generic repeated advice.

JSON schema:
{{
  "summary": "one short easy English sentence, max 18 words",
  "points": [
    "easy field sentence with crop/risk/input context, max 18 words",
    "easy action sentence for this exact risk level, max 18 words",
    "easy final sentence mentioning local weather or expert check, max 18 words"
  ]
}}

Inputs:
N={input_data["N"]}, P={input_data["P"]}, K={input_data["K"]}, temperature={input_data["temperature"]}, humidity={input_data["humidity"]}, ph={input_data["ph"]}, rainfall={input_data["rainfall"]}, area={input_data["area"]}, year={input_data["year"]}, pesticides_tonnes={input_data["pesticides_tonnes"]}

ML prediction:
recommended_crop={crop}
yield_risk={yield_risk}
final_decision={decision}
""".strip()

    payload = {  # Request body sent to Groq chat completion API.
        "model": os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You produce clean JSON agricultural advice from fixed ML outputs. "
                    "Never alter the provided crop, risk, or decision. Never use Markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 220,
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
        content = response_data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(content)
        summary = _strip_markdown_advice(parsed.get("summary", ""))
        points = [
            _strip_markdown_advice(point)
            for point in parsed.get("points", [])
            if _strip_markdown_advice(point)
        ][:3]
        if not summary:
            return fallback_parts, False
        if not points:
            points = fallback_parts["points"]
        return {"summary": summary, "points": points}, True
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        KeyError,
        IndexError,
        json.JSONDecodeError,
        TimeoutError,
        TypeError,
    ):
        return fallback_parts, False

# Main end-to-end function used by FastAPI and test_pipeline.py.
def predict_full_pipeline(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run Model 1, feed its crop into Model 2, and return the final result."""
    _validate_full_input(input_data)  # Stop early if user input is missing or invalid.

    crop_result = predict_crop(input_data)  # Model 1 predicts the best crop.
    recommended_crop = crop_result["recommended_crop"]  # This crop becomes Model 2 item input.

    risk_input = {  # Build the base input dictionary for Model 2.
        "area": input_data["area"],
        "item": recommended_crop,
        "year": input_data["year"],
        # Keep the form simple: one rainfall/temperature entry feeds both models.
        "average_rain_fall_mm_per_year": input_data["rainfall"],
        "pesticides_tonnes": input_data["pesticides_tonnes"],
        "avg_temp": input_data["temperature"],
    }

    risk_result = predict_yield_risk(risk_input)  # Model 2 predicts yield risk for recommended crop.
    yield_risk = risk_result["yield_risk"]  # LOW, MEDIUM, or HIGH.
    decision = make_decision(yield_risk)  # Convert risk into GO, CAUTION, or HOLD.
    crop_probabilities = crop_result["crop_probabilities"]
    risk_probabilities = risk_result["risk_probabilities"]
    top_3_predictions = crop_result["top_3_predictions"]
    crop_confidence = top_3_predictions[0]["confidence"] if top_3_predictions else 0.0
    crop_confidence_display = (
        top_3_predictions[0]["confidence_display"] if top_3_predictions else "0.00"
    )
    risk_probability_items = _probability_items(risk_probabilities)
    crop_risk_analysis = []  # Stores risk result for each top crop prediction.
    for crop_item in top_3_predictions:  # Check Model 2 risk for each of the top crop options.
        analysis_risk_input = {  # Build the base input dictionary for Model 2.
            "area": input_data["area"],
            "item": crop_item["crop"],
            "year": input_data["year"],
            "average_rain_fall_mm_per_year": input_data["rainfall"],
            "pesticides_tonnes": input_data["pesticides_tonnes"],
            "avg_temp": input_data["temperature"],
        }
        analysis_risk_result = predict_yield_risk(analysis_risk_input)  # Risk prediction for this top crop.
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

    fallback_advice = _make_advice(recommended_crop, yield_risk, decision)  # Always available backup advice.
    advice_parts, llm_advice_used = _make_llm_advice(  # Try Groq advice; fallback if unavailable.
        input_data=input_data,
        crop=recommended_crop,
        yield_risk=yield_risk,
        decision=decision,
        fallback_advice=fallback_advice,
    )
    advice = _format_advice_from_parts(advice_parts)
    advice_items = _advice_items(advice_parts["points"])

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
        "final_advice_summary": advice_parts["summary"],
        "final_advice_points": advice_parts["points"],
        "final_advice_items": advice_items,
        "llm_advice_used": llm_advice_used,
    }


# Public wrapper kept so frontend/tests can call run_pipeline(input_data).
def run_pipeline(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible Flask entry point."""
    return predict_full_pipeline(input_data)


def _percent(value: Any) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "N/A"


# Prints a VS Code terminal report for test_pipeline.py and direct pipeline runs.
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


# Allows running this file directly: python pipeline.py
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
