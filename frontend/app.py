"""FastAPI web app for the Crop Health & Yield Risk Predictor.

This file only handles the website layer: it receives form values from the
browser, sends them to pipeline.py, and renders the HTML result page.
"""

import sys  # Gives access to Python system settings, including import paths.
from pathlib import Path  # Helps build safe file/folder paths on Windows/Linux.

from fastapi import FastAPI, Form, Request  # FastAPI app, form input, and request object.
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


@app.get("/", response_class=HTMLResponse)  # GET route for opening the home/input page.
async def home(request: Request):  # request is required by Jinja2Templates.
    return templates.TemplateResponse(  # Renders and returns index.html to the browser.
        request,  # Passes current browser request to the template engine.
        "index.html",  # Template file shown for the input form.
        {},  # No extra data is needed on the input page.
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
    input_data = {  # Convert form values into the exact dictionary expected by pipeline.py.
        "N": float(N),  # Store Nitrogen as a numeric value.
        "P": float(P),  # Store Phosphorus as a numeric value.
        "K": float(K),  # Store Potassium as a numeric value.
        "temperature": float(temperature),  # Store temperature as a numeric value.
        "humidity": float(humidity),  # Store humidity as a numeric value.
        "ph": float(ph),  # Store pH as a numeric value.
        "rainfall": float(rainfall),  # Store rainfall as a numeric value.
        "area": area.strip(),  # Remove extra spaces from country/area text.
        "year": int(year),  # Store year as an integer.
        "pesticides_tonnes": float(pesticides_tonnes),  # Store pesticide usage as a numeric value.
    }

    result = run_pipeline(input_data)  # Runs Model 1, Model 2, decision logic, and advice generation.

    return templates.TemplateResponse(  # Renders the result dashboard page.
        request,  # Passes current request to Jinja.
        "result.html",  # Template file for prediction output.
        {
            "result": result,  # ML output dictionary used by result.html.
            "input_data": input_data,  # Original submitted values shown in the summary section.
        },
    )