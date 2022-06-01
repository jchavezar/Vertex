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

## Step 1: Building code, container and test them locally.

Set your variables:

```
REGION = [your_region]
BUCKET = [your_bucket]
```

Build a webserver docker container to handle predictions; uvicorn

```
CAT << EOF > Dockerfile
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY app /app
WORKDIR /app
RUN pip install sklearn joblib pandas tensorflow
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

EXPOSE 8080
EOF
```

Create code (logic) behind the webserver

```
if [ ! -d app ]; then
   mkdir app;
fi
```

```
CAT << EOF > app/main.py
from fastapi import Request, FastAPI
from tensorflow import keras
import json
import os

app = FastAPI()
BUCKET = 'gs://vertexlooker-models-central/mpg/model'
model = keras.models.load_model($BUCKET)


@app.get('/')
def get_root():
    return {'message': 'Welcome to the spam detection API: miles per gallon prediction'}


@app.get('/health_check')
def health():
    return 200


if os.environ.get('AIP_PREDICT_ROUTE') is not None:
    method = os.environ['AIP_PREDICT_ROUTE']
else:
    method = '/predict'


@app.post(method)
async def predict(request: Request):
    print("----------------- PREDICTING -----------------")
    body = await request.json()
    instances = body["instances"]
    outputs = model.predict(instances)
    response = outputs.tolist()
    print("----------------- OUTPUTS -----------------")

    return {"predictions": response}
EOF
```

Create a new repository in Google Cloud Platform to store containers. **(Remember to change `[YOUR_REGION]`)**

```
$gcloud artifacts repositories create repo-models --repository-format=docker \
--location=[YOUR_REGION] --description="Models repository"
```

Tag container in Artifacts repository format: **(Remember to change `[YOUR_REGION]` and `[YOUR_PROJECT]`)**
```
$docker build -t [YOUR_REGION]-docker.pkg.dev/['YOUR_PROJECT']/repo-models/container_model_test .
```

The easiest and secured way to handle GCP credentials is by using the Application Default Credentials, you have to login to get a temporary credentials:

```
$gcloud auth application-default login
```

This will generate a json config file with temporary credentials under: ~/.config/gcloud/, the container has to be able to mount that file through docker volumes, so let's define a variable that will be used when you run the container: **(Remember to change the `[USERNAME]`)**

```
$ADC=/home/[USERNAME]/.config/gcloud/application_default_credentials.json
```

Run the container locally **(Remember to change `[YOUR_REGION]`)**:
```
$docker run --name predict \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/FILE_NAME.json \
  -v ${ADC}:/tmp/keys/FILE_NAME.json \
  -p 732:8080 [YOUR_REGION]-docker.pkg.dev/[YOUR_PROJECT]/repo-models/container_model_test
```

You can break it down with Ctrl+C.

For predictions, open a new terminal an make an http request with the data in json format:

```
curl -X POST -H "Content-Type: application/json" http://localhost:732/predict -d '{
 "instances": [[1.4838871833555929,
 1.8659883497083019,
 2.234620276849616,
 1.0187816540094903,
 -2.530890710602246,
 -1.6046416850441676,
 -0.4651483719733302,
 -0.4952254087173721,
 0.7746763768735953]]
}'
```


