import sys
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import run_pipeline  # noqa: E402


app = FastAPI(title="Crop Health & Yield Risk Predictor")

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=FRONTEND_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {},
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    N: float = Form(...),
    P: float = Form(...),
    K: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...),
    area: str = Form(...),
    year: int = Form(...),
    pesticides_tonnes: float = Form(...),
):
    input_data = {
        "N": float(N),
        "P": float(P),
        "K": float(K),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "ph": float(ph),
        "rainfall": float(rainfall),
        "area": area.strip(),
        "year": int(year),
        "pesticides_tonnes": float(pesticides_tonnes),
    }

    result = run_pipeline(input_data)

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "result": result,
            "input_data": input_data,
        },
    )
