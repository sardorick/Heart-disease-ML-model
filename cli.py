import requests
import json
import argparse
import joblib
import pandas as pd
import time 
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

 
test_df = pd.read_csv('test.csv')


# Loading model
model = joblib.load('model.pkl')

# Setting up the CLI
parser = argparse.ArgumentParser(description= 'Heart attack predictor')

parser.add_argument('test_data', type=str, help='input data')

args = parser.parse_args()

test_data_link = args.test_data

# Loading records
input_data = pd.read_csv(test_data_link)
column_names = input_data.columns

# Predictions
for _, row in input.iterrows():
    x = pd.DataFrame([row])

    print(x.to_string(index=False), '\n')

    preds = model.predict(x)[0]
    if preds == 1:
        print("Heart attack probability")
    else:
        print("No risk of heart attack")
    print('-'*100, '\n\n')

    # Pause (to simulate readings every 3 secondes)
    # time.sleep(3)
