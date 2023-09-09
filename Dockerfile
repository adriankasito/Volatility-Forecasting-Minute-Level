FROM python:3.9


RUN pip install fastapi uvicorn
# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

COPY main.py /app/

CMD uvicorn main:app --host 0.0.0.0 --port 8000
