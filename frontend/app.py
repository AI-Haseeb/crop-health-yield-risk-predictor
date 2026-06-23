"""FastAPI web app for the Crop Health & Yield Risk Predictor.

This file only handles the website layer: it receives form values from the
browser, sends them to pipeline.py, and renders the HTML result page.
"""

import sys  # Gives access to Python system settings, including import paths.
import json  # Reads JSON responses from the weather API.
import urllib.parse  # Safely adds city/country text into API URLs.
import urllib.request  # Calls the free weather API from the backend.
from datetime import datetime  # Gets the current year for auto-fill.
from pathlib import Path  # Helps build safe file/folder paths on Windows/Linux.

from fastapi import Body, FastAPI, Form, HTTPException, Query, Request  # FastAPI app, form input, API errors, query input, and request object.
from fastapi.responses import HTMLResponse  # Tells FastAPI that routes return HTML pages.
from fastapi.staticfiles import StaticFiles  # Serves CSS/images/JS from the static folder.
from fastapi.templating import Jinja2Templates  # Connects FastAPI with Jinja HTML templates.


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Main project folder containing pipeline.py.
FRONTEND_DIR = Path(__file__).resolve().parent  # frontend folder containing templates and static files.
sys.path.insert(0, str(PROJECT_ROOT))  # Adds project root so Python can import pipeline.py.

from pipeline import run_pipeline  # noqa: E402  # Imports the final ML pipeline function after path setup.


app = FastAPI(title="Crop Health & Yield Risk Predictor")  # Creates the FastAPI web application object.

app.mount(  # Makes frontend/static available in the browser as /static.
    "/static",  # URL prefix used inside HTML/CSS links.
    StaticFiles(directory=FRONTEND_DIR / "static"),  # Actual folder where CSS and assets are stored.
    name="static",  # Name used by url_for('static', ...) in templates.
)

templates = Jinja2Templates(directory=FRONTEND_DIR / "templates")  # Folder containing index.html/result.html.

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"  # Converts city/country text into latitude/longitude.
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"  # Gets current weather values for those coordinates.


def _read_json_url(url: str) -> dict:  # Small helper that opens a URL and returns parsed JSON.
    with urllib.request.urlopen(url, timeout=8) as response:  # Timeout keeps the app from waiting too long.
        return json.loads(response.read().decode("utf-8"))  # Decode API response into a Python dictionary.


@app.get("/weather")  # Lightweight API route used by the frontend Auto Weather button.
async def weather_lookup(location: str = Query(..., min_length=2)):  # location can be city, country, or both.
    clean_location = location.strip()  # Remove extra spaces from user text.
    geocoding_query = urllib.parse.urlencode(  # Build safe URL query parameters.
        {"name": clean_location, "count": 1, "language": "en", "format": "json"}
    )

    try:
        geo_data = _read_json_url(f"{GEOCODING_URL}?{geocoding_query}")  # Find coordinates for the location.
        locations = geo_data.get("results") or []  # Open-Meteo returns matching places in results.
        if not locations:
            raise HTTPException(status_code=404, detail="Location not found. Try city + country.")

        place = locations[0]  # Use the best match returned by geocoding.
        latitude = place["latitude"]  # Latitude needed by weather forecast API.
        longitude = place["longitude"]  # Longitude needed by weather forecast API.
        weather_query = urllib.parse.urlencode(  # Ask for current weather and today precipitation.
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation,rain",
                "daily": "precipitation_sum",
                "timezone": "auto",
                "temperature_unit": "celsius",
                "precipitation_unit": "mm",
                "forecast_days": 1,
            }
        )
        weather_data = _read_json_url(f"{WEATHER_URL}?{weather_query}")  # Fetch weather for the coordinates.
        current = weather_data.get("current", {})  # Current weather block from API response.
        daily = weather_data.get("daily", {})  # Daily forecast block from API response.
        daily_rainfall = (daily.get("precipitation_sum") or [current.get("precipitation", 0)])[0]

        return {  # Send clean weather values back to the browser.
            "location_name": place.get("name", clean_location),
            "country": place.get("country", ""),
            "latitude": latitude,
            "longitude": longitude,
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "rainfall": daily_rainfall,
            "current_precipitation": current.get("precipitation"),
            "year": datetime.now().year,
            "source": "Open-Meteo",
            "note": "Rainfall is today forecast/suggestion in mm. Edit it if you have seasonal or annual rainfall.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weather lookup failed: {exc}") from exc





def _clean_pipeline_input(raw_data: dict) -> dict:
    """Convert browser/API input into the exact dictionary expected by pipeline.py."""
    try:
        return {
            "N": float(raw_data["N"]),
            "P": float(raw_data["P"]),
            "K": float(raw_data["K"]),
            "temperature": float(raw_data["temperature"]),
            "humidity": float(raw_data["humidity"]),
            "ph": float(raw_data["ph"]),
            "rainfall": float(raw_data["rainfall"]),
            "area": str(raw_data["area"]).strip(),
            "year": int(raw_data["year"]),
            "pesticides_tonnes": float(raw_data["pesticides_tonnes"]),
        }
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing input field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid numeric input: {exc}") from exc


@app.post("/api/predict")
async def predict_api(input_payload: dict = Body(...)):
    """Return JSON prediction for dynamic city-wise comparison cards."""
    input_data = _clean_pipeline_input(input_payload)
    return run_pipeline(input_data)

@app.get("/", response_class=HTMLResponse)  # GET route for opening the home/input page.
async def home(request: Request):  # request is required by Jinja2Templates.
    return templates.TemplateResponse(  # Renders and returns index.html to the browser.
        request,  # Passes current browser request to the template engine.
        "index.html",  # Template file shown for the input form.
        {"current_year": datetime.now().year},  # Sends current year so the form can show it by default.
    )


@app.post("/predict", response_class=HTMLResponse)  # POST route called when the form is submitted.
async def predict(
    request: Request,  # Current request object needed for rendering Jinja templates.
    N: float = Form(...),  # Nitrogen value from the form.
    P: float = Form(...),  # Phosphorus value from the form.
    K: float = Form(...),  # Potassium value from the form.
    temperature: float = Form(...),  # Temperature value; also goes to Model 2 as avg_temp.
    humidity: float = Form(...),  # Humidity value for crop recommendation.
    ph: float = Form(...),  # Soil pH value for crop recommendation.
    rainfall: float = Form(...),  # Rainfall value; also goes to Model 2 as annual rainfall.
    area: str = Form(...),  # Country/area value used by the yield risk model.
    year: int = Form(...),  # Target year used by the yield risk model.
    pesticides_tonnes: float = Form(...),  # Pesticide usage used by the yield risk model.
):
    input_data = _clean_pipeline_input({  # Convert form values into the exact dictionary expected by pipeline.py.
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
        "area": area,
        "year": year,
        "pesticides_tonnes": pesticides_tonnes,
    })

    result = run_pipeline(input_data)  # Runs Model 1, Model 2, decision logic, and advice generation.

    return templates.TemplateResponse(  # Renders the result dashboard page.
        request,  # Passes current request to Jinja.
        "result.html",  # Template file for prediction output.
        {
            "result": result,  # ML output dictionary used by result.html.
            "input_data": input_data,  # Original submitted values shown in the summary section.
        },
    )