import sqlite3
import os
from config import settings
from data import SQLRepository
from fastapi import FastAPI
from fastapi.responses import FileResponse 
from model import GarchModel
from pydantic import BaseModel



class FitIn(BaseModel):
    ticker:str
    use_new_data: bool
    n_observations: int
    p: int
    q: int


class FitOut(FitIn):
    success: bool
    message: str
    


class PredictIn(BaseModel):
    ticker: str
    n_minutes: int


class PredictOut(PredictIn):
    success: bool
    forecast: dict
    message: str


def build_model(ticker, use_new_data):

    # Create DB connection
    connection = sqlite3.connect(settings.db_name, check_same_thread=False)

    #Create `SQLRepository`
    repo = SQLRepository(connection=connection)

    # Create model
    model = GarchModel(ticker=ticker, use_new_data=use_new_data, repo=repo)

    #Return model
    return model


app = FastAPI()


@app.get("/", status_code=200)
def hello():
    """Return dictionary with greeting message."""
    return {"message": "Welcome to the Volatility Forecasting Project!"}



@app.post("/fit", status_code=200, response_model=FitOut)
def fit_model(request: FitIn):

    """Fit model, return confirmation message.

    Parameters
    ----------
    request : FitIn

    Returns
    ------
    dict
        Must conform to `FitOut` class
    """
    # Create `response` dictionary from `request`
    response = request.dict()


    # Create try block to handle exceptions
    try:

        # Build model with `build_model` function
        model = build_model(ticker = request.ticker, use_new_data=request.use_new_data)
        

        # Wrangle data
        model.wrangle_data(n_observations = request.n_observations)


        # Fit model
        model.fit(p=request.p, q=request.q)


        # Save model
        filename= model.dump()
        
        # get AIC and BIC scores from the fitted model 
        #aic = model.aic
        #bic = model.bic


        # Add `"success"` key to `response`
        response["success"] = True


        # Add `"message"` key to `response` with `filename`
        response["message"] = f"Trained and saved {filename}. Metrics: AIC {model.aic}, BIC {model.bic}."
        

    # Create except block
    except Exception as e:

        # Add `"success"` key to `response`
        response["success"] = False


        # Add `"message"` key to `response` with error message
        response["message"] = str(e)


    # Return response
    return response


@app.post("/predict", status_code=200, response_model=PredictOut)
def get_prediction(request: PredictIn):

    # Create `response` dictionary from `request`
    #response = request.dict()
    response = request


    # Create try block to handle exceptions
    try:

        # Build model with `build_model` function
        model = build_model(ticker = request.ticker, use_new_data=False)


        # Load stored model
        model.load()


        # Generate prediction
        prediction = model.predict_volatility(horizon = request.n_minutes)


        # Add `"success"` key to `response`
        response["success"] = True


        # Add `"forecast"` key to `response`
        response["forecast"] = prediction


        # Add `"message"` key to `response`
        response["message"] = "Prediction Successful"


    # Create except block
    except Exception as e:

        # Add `"success"` key to `response`
        response["success"] = False


        # Add `"forecast"` key to `response`
        response["forecast"] = {}


        #  Add `"message"` key to `response`
        response["message"] = str(e)


    # Return response
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

