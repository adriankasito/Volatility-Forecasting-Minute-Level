import sqlite3
from fastapi import FastAPI
from starlette.responses import HTMLResponse
from config import settings
from data import SQLRepository
from model import GarchModel
from pydantic import BaseModel

class FitIn(BaseModel):
    ticker: str
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
    connection = sqlite3.connect(settings.db_name, check_same_thread=False)
    repo = SQLRepository(connection=connection)
    model = GarchModel(ticker=ticker, use_new_data=use_new_data, repo=repo)
    return model

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get_volatility_forecasts():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Volatility Forecast</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                text-align: center;
            }
            h1 {
                color: #333;
            }
            .container {
                background-color: #fff;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
                padding: 20px;
                margin: 20px;
            }
            button {
                background-color: #007bff;
                color: #fff;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
            #minutesSlider {
                width: 80%;
                margin: 10px auto;
            }
            table {
                width: 80%;
                margin: 20px auto;
                border-collapse: collapse;
                text-align: center;
            }
            th, td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            .odd {
                background-color: #f5f5f5;
            }
            .even {
                background-color: #e0e0e0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Volatility Forecasting In Microsoft Corporation</h1>
            <p>Minute-level volatility forecasting for Microsoft involves predicting short-term price fluctuations in Microsoft Corporation's stock, utilizing historical data and statistical models. Traders and investors seek to leverage this information for various purposes, including executing precise short-term trading strategies, managing risk exposure, optimizing algorithmic trading algorithms, and making informed decisions in options trading and market timing. It aids in achieving more accurate entry and exit points and enhances portfolio optimization, making it a valuable tool for those navigating the dynamic landscape of financial markets and aiming to maximize returns while minimizing risk.</p>
            <label for="minutesSlider">Time range in minutes:</label>
            <input type="range" min="1" max="10000" value="25" class="slider" id="minutesSlider">
            <p id="selectedMinutes">Selected minutes: 25</p>
            <button id="forecastButton">Get Volatility Forecast</button>
            <table>
                <tr>
                    <th>Microsoft Corporation Volatility By Minute</th>
                    <th>Date</th>
                    <th>Time</th>
                    <th>Volatility</th>
                </tr>
            </table>
        </div>
        <script>
            document.getElementById("minutesSlider").addEventListener("input", function() {
                document.getElementById("selectedMinutes").textContent = "Selected minutes: " + this.value;
            });

            document.getElementById("forecastButton").addEventListener("click", function() {
                const selectedMinutes = document.getElementById("minutesSlider").value;
                fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({"ticker": "MSFT", "n_minutes": selectedMinutes}),
                })
                .then(response => response.json())
                .then(data => {
                    const table = document.querySelector("table");
                    table.innerHTML = `
                        <tr>
                            <th>MSFT</th>
                            <th>Date</th>
                            <th>Time</th>
                            <th>Volatility</th>
                        </tr>`;
                    let rowNumber = 1;
                    for (const [timestamp, volatility] of Object.entries(data.forecast)) {
                        const [date, time] = timestamp.split("T");
                        const className = rowNumber % 2 === 0 ? "even" : "odd";
                        table.innerHTML += `
                            <tr class="${className}">
                                <td>${rowNumber}</td>
                                <td>${date}</td>
                                <td>${time}</td>
                                <td>${volatility}</td>
                            </tr>`;
                        rowNumber++;
                    }
                })
                .catch(error => {
                    const table = document.querySelector("table");
                    table.innerHTML = `<tr><td colspan="4">Error: ${error.message}</td></tr>`;
                });
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/fit", status_code=200, response_model=FitOut)
def fit_model(request: FitIn):
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
def get_prediction(request: PredictIn):
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
