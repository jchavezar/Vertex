# Vertex AI Custom Container Deployment

**Snippets for Google Vertex AI!**

---

You can:

1. Use your own pc/laptop for locally testing.
2. User any flavor of Vertex Workbench [here](https://console.cloud.google.com/vertex-ai/workbench).
3. Upload and deploy it on Vertex Endpoints [here](https://console.cloud.google.com/vertex-ai/endpoints).
4. Mix between 1.+3. or 2.+3.

## Data Type

- request in json format ({'instances' : [1,2,3,4]})

---

## Authentication

## Step 1: Building code and container and locally testing.

Build a docker container with a webserver for predictions; uvicorn

Dockerfile
```
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY app /app
WORKDIR /app
RUN pip install sklearn joblib pandas tensorflow
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

EXPOSE 8080
```