import os
import sqlite3
from fastapi import FastAPI
from starlette.responses import HTMLResponse
from config import settings
from data import SQLRepository
from model import GarchModel
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

import json

class FitIn(BaseModel):
    """Data model for fitting GARCH model parameters"""
    ticker: str
    use_new_data: bool
    n_observations: int
    p: int
    q: int


class FitOut(FitIn):
    """Response data model for fitting GARCH model parameters"""
    success: bool
    message: str


class PredictIn(BaseModel):
    """Data model for predicting volatility"""
    ticker: str
    n_minutes: int


class PredictOut(PredictIn):
    """Response data model for predicting volatility"""
    success: bool
    forecast: dict
    message: str


def build_model(ticker: str, use_new_data: bool) -> GarchModel:
    """Builds a GARCH model for the specified ticker and data usage"""
    connection = sqlite3.connect(settings.db_name)
    repo = SQLRepository(connection=connection)
    model = GarchModel(ticker=ticker, use_new_data=use_new_data, repo=repo)
    return model


app = FastAPI()

app.mount("/static", StaticFiles(directory="docs/static"), name="static")
html_file_path = os.path.join(os.path.dirname(__file__), "docs", "index.html")

@app.get("/")
def get_volatility_forecasts() -> HTMLResponse:
    """Renders the HTML page for volatility forecasting"""
    try:
        with open(os.path.join(os.path.dirname(__file__), "docs", "index.html"), "r") as html_file:
            html_content = html_file.read()

        return HTMLResponse(content=html_content, status_code=200, headers={"Content-Type": "text/html"})

    except Exception as e:
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500, headers={"Content-Type": "text/html"})


@app.post("/fit", status_code=200, response_model=FitOut)
def fit_model(request: FitIn) -> FitOut:
    """Fits the GARCH model using the specified parameters"""
    response = request.dict()
    try:
        model = build_model(ticker=request.ticker, use_new_data=request.use_new_data)
        model.wrangle_data(n_observations=request.n_observations)
        model.fit(p=request.p, q=request.q)
        filename = model.dump()

        response["success"] = True
        response["message"] = f"Trained and saved {filename}. Metrics: AIC {model.aic}, BIC {model.bic}."

    except Exception as e:
        response["success"] = False
        response["message"] = str(e)
    return response


@app.post("/predict", status_code=200, response_model=PredictOut)
def get_prediction(request: PredictIn) -> PredictOut:
    """Predicts volatility for the specified time range"""
    response = request.dict()
    try:
        model = build_model(ticker=request.ticker, use_new_data=False)
        model.load()
        prediction = model.predict_volatility(horizon=request.n_minutes)

        response["success"] = True
        response["forecast"] = prediction
        response["message"] = ""

    except Exception as e:
        response["success"] = False
        response["forecast"] = {}
        response["message"] = str(e)
    return response

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8008)
