FROM python:3.9

WORKDIR /app

RUN pip install fastapi uvicorn


COPY main.py model.py data.py config.py VolatilityForecastingProject.py requirements.txt ./


RUN pip install -r requirements.txt

# Copy your application code into the container
COPY . .


CMD uvicorn main:app --host 0.0.0.0 --port 8080

EXPOSE 8080
