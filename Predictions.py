from flask.wrappers import Response
import joblib
from flask import Flask
from flask import request
from flask import json
import traceback
import numpy as np
import pandas as pd

BUCKET_NAME = 'disease-prediction-model'
KEY = 'finalized_model.pkl'
MODEL_LOCAL_PATH = KEY

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
  response = json.loads(request.get_data().decode('utf-8'))
  prediction = predict(response)
  print(prediction)
  data = {}
  data['disease'] = prediction[-1]
  return json.dumps(data)

def load_model():
  
  #conn = S3Connection(aws_access_key_id='AKIAYZVN6JE7LXYBK5PO' , aws_secret_access_key= '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', host='s3.us-east-2.amazonaws.com')
  #bucket = conn.get_bucket(BUCKET_NAME)
  #key_obj = Key(bucket)
  #key_obj.key = KEY

  #contents = key_obj.get_contents_to_filename(MODEL_LOCAL_PATH)
 
  return joblib.load('finalized_model.pkl')

def predict(data):
  print('Prediction of disease')
  # Process your data, create a dataframe/vector and make your predictions
  
  psymptoms = pd.DataFrame(data['symptoms'])


  #conn = S3Connection(aws_access_key_id='access_id' , aws_secret_access_key= 'secret_access', host='host_name')
  #bucket = conn.get_bucket(BUCKET_NAME)
  #key_obj = Key(bucket)
  #key_obj.key = 'Symptom-severity.csv'

  #contents = key_obj.get_contents_to_filename('Symptom-severity.csv')
 
  severity_df = pd.read_csv('Symptom-severity.csv')

  a = np.array(severity_df["Symptom"])
  b = np.array(severity_df["weight"])
  psymptoms = np.array(psymptoms)
  for j in range(len(psymptoms)):
    for k in range(len(a)):
        if psymptoms[j]==a[k]:
          psymptoms[j]=b[k]


  psymptoms = pd.DataFrame(psymptoms)
  psymptoms = psymptoms.replace('dischromic__patches', 0)
  psymptoms = psymptoms.replace('spotting__urination',0)
  psymptoms = psymptoms.replace('foul_smell_of_urine',0)

  #Append zero for extra symptoms
  max_symptoms = 17
  nulls = [0] * (max_symptoms - len(psymptoms))
  #psy = [psymptoms.flatten() + nulls.flatten()]
  psy = np.append(psymptoms, nulls)
  final_formatted_data = [psy]
  print('loading model')
  predicted_disease =  load_model().predict(final_formatted_data)
  return predicted_disease



if __name__ == '__main__':
  app.run(host='0.0.0.0', port = 8080)
