FROM python:3.9

RUN pip install fastapi uvicorn

COPY main.py /app/

CMD uvicorn main:app --host 0.0.0.0 --port 8000
