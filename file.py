import pandas as pd 
age = [int(input("How old are you? \n"))]
sex = [int(input("What is your gender? 1 for male and 0 for female \n"))]
cp = [int(input("Which type of chest pain do you have? [0, 1, 2, 3] \n"))]
trtbps = [int(input("What is your resting blood pressure? \n"))]
chol = [int(input("What is your cholesterol level? \n"))]
fbs = [int(input("Is your fasting blood sugar higher than 120 mg/dl?1 for yes and 0 for no \n"))]
restecg = [int(input("What are your resting ECG? [0,1,2] \n"))]
thalach = [int(input("What is your maximum heartrate? \n"))]
exang = [int(input("Do you have EXang? 1 for yes and 0 for no\n"))]
oldpeak = [float(input("What is your oldpeak value? answer should be float\n"))]
slp = [int(input("The slope of your peak exercise segment: [0, 1, 2] \n"))]
caa = [int(input("The number of your vessels: [0-3] \n"))]
thal = [int(input("Thal value: [1, 2, 3] \n"))]

list = [age,sex,cp,trtbps,chol,fbs,restecg,thalach,exang,oldpeak,slp,caa,thal]
dict = {'List': list}
df = pd.DataFrame(dict)
df.to_csv('file.csv')