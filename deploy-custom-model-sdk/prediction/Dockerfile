FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY app /app
WORKDIR /app
RUN pip install sklearn joblib pandas tensorflow
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

EXPOSE 8080
