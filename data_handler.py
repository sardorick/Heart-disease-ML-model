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



data = pd.read_csv(r'data\heart.csv') 



x = data.drop(['output'], axis=1) # features - train and val data
y = data['output'] # target

num_vals = ['age', 'trtbps','thalachh', 'chol', 'oldpeak']
cat_vals = ['sex', 'cp', 'exng', 'slp', 'caa', 'thall', 'restecg', 'fbs']
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)


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
    "Extra Trees": ExtraTreesClassifier(random_state=0, n_estimators=100),
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

    model.fit(x_train, y_train)

    predics = model.predict(x_val)
    total_time = time.time() - start_time
    


    results = results.append({"Model": model_name,
                            "Accuracy Score": accuracy_score(y_val, predics)*100,
                            "Balanced Accuracy score": balanced_accuracy_score(y_val, predics)*100,
                            "Time": total_time}, ignore_index=True)

results_order = results.sort_values(by=['Accuracy Score'], ascending=False, ignore_index=True)

print(results_order)

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