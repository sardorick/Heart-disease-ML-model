import requests
import json
import argparse
import pandas as pd 
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from data_handler import predictor, x_train, y_train, x_val, y_val

 
test_df = pd.read_csv('test.csv')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description= 'Heart attack predictor')

    parser.add_argument('test', type=str, help='input data')
    # parser.add_argument('sex', type=int, choices=[0, 1] , help='Male = 1, Female = 0')
    # parser.add_argument('cp', type=int, choices=[1, 2, 3, 4] , help='Chest Pain type , between 1 and 4')

    # parser.add_argument('trtbps', type=int, help='resting blood pressure (in mm Hg)')
    # parser.add_argument('chol', type=int, help='cholestoral in mg/dl fetched via BMI sensor')
    # parser.add_argument('fbs', type=int, choices=[0, 1] , help='fasting blood sugar > 120 mg/dl) (1 = true; 0 = false')

    # parser.add_argument('restecg', type=int, choices=[0, 1, 2] , help='resting electrocardiographic results, between 0 and 2')
    # parser.add_argument('thalachh', type=int, help='maximum heart rate achieved')
    # parser.add_argument('exng', type=int, choices=[0, 1] , help='exercise induced angina (1 = yes; 0 = no)')

    # parser.add_argument('oldpeak', type=float, help='previous peak')
    # parser.add_argument('slp', type=int, choices=[0, 1, 2] , help='slope, between 0 and 2')
    # parser.add_argument('caa', type=int, choices=[0, 1, 2, 3] , help='number of major vessels (0-3)')
    # parser.add_argument('thall', type=int, choices=[0, 1, 2, 3] , help='Thall rate, between 0 and 3')
    args = parser.parse_args()
    
    # args_list = [args.age, args.sex, args.cp, args.trtbps, args.chol, args.fbs, args.restecg, args.thalachh, args.exng, args.oldpeak, 
    #             args.slp, args.caa, args.thall]
    # args_list = np.array(args_list)
    # args_list = args_list.reshape((1, -1))
    args_input = args.test_df
    # args_input = np.array(args_input)
    # args_input = args_input.reshape((1, -1))

    pred = predictor(args_input)
    print(pred)

    

    # def get_inputs():
    #     input_features = []
    #     age = int(input("How old are you? \n"))
    #     sex = int(input("Gender? 0 for Female, 1 for Male \n"))
    #     cp = int(input("Chest pain type? 0 for Absent, 1 for light pain, 2 for moderate pain, 3 for extreme pain \n"))
    #     trtbps = int(input("Resting blood pressure in mm Hg \n"))
    #     chol = int(input("Serum cholestrol in mg/dl \n"))
    #     fbs = int(input("Fasting Blood Sugar? 0 for < 120 mg/dl, 1 for > 120 mg/dl \n"))
    #     restecg = int(input("Resting ecg? (0,1,2) \n"))
    #     thalachh = int(input("Maximum Heart Rate achieved? \n"))
    #     exng = int(input("Exercise Induced Angina? 0 for no, 1 for yes \n"))
    #     oldpeak = float(input("Old Peak? ST Depression induced by exercise relative to rest \n"))
    #     slp = int(input("Slope of the peak? (0,1,2) \n"))
    #     caa = int(input("Number of colored vessels during Floroscopy? (0,1,2,3) \n"))
    #     thall = int(input("thal: 3 = normal; 6 = fixed defect; 7 = reversable defect \n"))
    #     input_features.append([age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall])
    #     df = pd.DataFrame(input_features, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall'])
    #     csv = df.to_csv('file.csv')
    #     return csv
