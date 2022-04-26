import argparse
import joblib
import pandas as pd
import numpy as np

 
# test_df = pd.read_csv('test.csv')


# Loading model
model = joblib.load('model.pkl')

parser = argparse.ArgumentParser(description= 'Heart attack predictor')

parser.add_argument('test_data', type=str, help='Insert a path for the input data')

args = parser.parse_args()

test_data_link = args.test_data

# data
input_data = pd.read_csv(test_data_link)
column_names = input_data.columns

# Predictions
for _, sample in input_data.iterrows():
    x = pd.DataFrame([sample])

    print(x.to_string(), '\n')

    preds = model.predict(x)[0]
    if preds == 1:
        print("You have heart attack probability")
    else:
        print("Your heart is healthy")
    print('-'*100, '\n\n')


