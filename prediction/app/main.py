from fastapi import Request, FastAPI
from tensorflow import keras
import json
import os

app = FastAPI()

if os.environ.get('AIP_STORAGE_URI') is not None:
    BUCKET = os.environ['AIP_STORAGE_URI']
else:
    BUCKET = 'gs://vertexlooker-central/mpg3/model'

model = keras.models.load_model(BUCKET)


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
