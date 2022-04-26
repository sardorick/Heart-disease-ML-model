from email import header
from operator import index
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
np.random.seed(0)



data = pd.read_csv(r'data\heart.csv')
data = data.drop(['chol', 'fbs'], axis=1) # dropping 'chol' and 'fbs' doesn't do any performance improvement and doesn't decrease performance.
# but dropping 'restecg' does. 

# creating a new feature - rate of resting blood pressure relative to maximum heart rate achieved
# data['pressure_heart'] = data['trtbps'] / data['thalachh'] 

# data['pressure_heart'] = data['thalachh'] / data['trtbps'] #adding this decreases the accuracy


# Build a data enhancer

def data_enhance(data):
    org_data = data
    for sex in data['sex'].unique():
        sex_data = org_data[org_data['sex']==sex]
        age_std = sex_data['age'].std()
        trtbps_std = sex_data['trtbps'].std()
        # chol_std = sex_data['chol'].std()
        thalachh_std = sex_data['thalachh'].std()
        oldpeak_std = sex_data['oldpeak'].std()
        for i in org_data[org_data['sex']==sex].index:
            if np.random.randint(2) == 1:
                org_data['age'].values[i] += age_std/10
                org_data['trtbps'].values[i] += trtbps_std/10
                # org_data['chol'].values[i] += chol_std/10
                org_data['thalachh'].values[i] += thalachh_std/10
                org_data['oldpeak'].values[i] += oldpeak_std/10
            else:
                org_data['age'].values[i] -= age_std/10
                org_data['trtbps'].values[i] -= trtbps_std/10
                # org_data['chol'].values[i] -= chol_std/10
                org_data['thalachh'].values[i] -= thalachh_std/10
                org_data['oldpeak'].values[i] -= oldpeak_std/10


    return org_data

gen = data_enhance(data)
gen = gen.sample(frac=1)
x = data.drop(['output'], axis=1) # features - train and val data
y = data['output'] # target

num_vals = ['age', 'trtbps','thalachh', 'oldpeak'] 
cat_vals = ['sex', 'cp', 'exng', 'slp', 'caa', 'thall', 'restecg'] 
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

# Add enhanced data to 20% of the orig data
enhanced_sample = gen.sample(gen.shape[0] // 5)
x_train_enh = pd.concat([x_train, enhanced_sample.drop(['output'], axis=1 ) ])
y_train_enh = pd.concat([y_train, enhanced_sample['output'] ])
print(f'Augmented percenteage {((len(x_train_enh) - len(x_train)) / len(x_train)) * 100} %')


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))])

tree_pipe = ColumnTransformer([('num', num_pipeline, num_vals), ('cat', cat_pipeline, cat_vals)], remainder='passthrough')

# Different classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0, n_estimators=100),
    "Ada Boost": AdaBoostClassifier(random_state=0, n_estimators=100),
    "Extra Trees": ExtraTreesClassifier(max_depth=9, min_samples_split=4, n_estimators=500, random_state=42), # improved 1% by GridsearchCV
    "Gradient Boosting": GradientBoostingClassifier(random_state=0, n_estimators=100),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(random_state=0, n_estimators=100),
    "Catboost": CatBoostClassifier(random_state=0, n_estimators=100),
    "Logistic Regression": LogisticRegression(random_state=0)
}

classifiers = {name: make_pipeline(tree_pipe, model) for name, model in classifiers.items()}

# Results df
results = pd.DataFrame({'Model': [], "Accuracy Score": [], "Balanced Accuracy score": [], "Time": []})



for model_name, model in classifiers.items():
    start_time = time.time()

    model.fit(x_train_enh, y_train_enh)

    predics = model.predict(x_val)
    total_time = time.time() - start_time
    


    results = results.append({"Model": model_name,
                            "Accuracy Score": accuracy_score(y_val, predics)*100,
                            "Balanced Accuracy score": balanced_accuracy_score(y_val, predics)*100,
                            "Time": total_time}, ignore_index=True)

results_order = results.sort_values(by=['Accuracy Score'], ascending=False, ignore_index=True)

# print(results_order)
import joblib
best_model = classifiers.get("Extra Trees")
joblib.dump(best_model, 'model.pkl')

# def trainer():
#     best_model = classifiers.get("Extra Trees")
#     best_model.fit(x_train, y_train)
#     return best_model
# location = 'C:\Users\Lenovo\OneDrive\Documents\Strive repos\Heart-disease-ML-model'
# def predictor(features):
#     preds = trainer().predict(features)
#     return preds




# create test data

df = data.sample(n=6)
df = df.drop('output', axis=1)
df.to_csv('test.csv')

# Benchmark
"""
                 Model  Accuracy Score  Balanced Accuracy score      Time
0        Random Forest       85.245902                84.858388  0.179506
1          Extra Trees       85.245902                84.477124  0.106714
2  Logistic Regression       85.245902                84.477124  0.051864
3            Ada Boost       83.606557                84.150327  0.161600
4             LightGBM       83.606557                83.769063  0.105717
5             Catboost       81.967213                81.535948  0.446212
6    Gradient Boosting       80.327869                80.065359  0.076796
7              XGBoost       78.688525                78.594771  0.131649
8        Decision Tree       75.409836                75.653595  0.012834
"""

# STD
"""
                 Model  Accuracy Score  Balanced Accuracy score      Time
0          Extra Trees       90.163934                89.651416  0.146654
1        Random Forest       88.524590                88.180828  0.162445
2             Catboost       88.524590                88.180828  0.670215
3            Ada Boost       86.885246                87.091503  0.181450
4    Gradient Boosting       86.885246                87.091503  0.094347
5              XGBoost       86.885246                87.472767  0.209441
6  Logistic Regression       86.885246                86.328976  0.115691
7        Decision Tree       85.245902                84.858388  0.011972
8             LightGBM       85.245902                85.239651  0.111703
"""

# With GridSearchCV
"""
                 Model  Accuracy Score  Balanced Accuracy score      Time
0          Extra Trees       93.442623                93.355120  0.541791
1        Random Forest       88.524590                88.180828  0.168255
2             Catboost       88.524590                88.180828  0.437876
3            Ada Boost       86.885246                87.091503  0.197509
4    Gradient Boosting       86.885246                87.091503  0.086766
5              XGBoost       86.885246                87.472767  0.133575
6  Logistic Regression       86.885246                86.328976  0.073803
7        Decision Tree       85.245902                84.858388  0.025932
8             LightGBM       85.245902                85.239651  0.093745
"""
# use joblib to save the best model and divide the predict function into two