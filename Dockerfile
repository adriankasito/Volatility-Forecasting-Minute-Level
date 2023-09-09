FROM python:3.9


RUN pip install fastapi uvicorn
# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

COPY app.py /app/

CMD uvicorn app:app --host 0.0.0.0 --port 8000

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["python", "app.py"]