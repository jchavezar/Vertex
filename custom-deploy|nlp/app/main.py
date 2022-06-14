import joblib
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, status
import json
from fastapi import Request, FastAPI
import pandas as pd
import os

app = FastAPI()

model = joblib.load('spam_classifier.joblib')


@app.get('/')
def get_root():
    return {'message': 'Welcome to the spam detection API: ham if you good, spam if you bad'}


@app.get('/health_check')
def health():
    return 200


@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    print("----------------- PREDICTING -----------------")
    body = await request.json()
    instances = body["instances"]
    column_headers = ['message']
    inputs = pd.DataFrame(instances, columns=column_headers)
    outputs = model.predict(inputs)
    print("----------------- OUTPUTS -----------------")
    print(outputs.tolist())

    return {"predictions": outputs.tolist()}
