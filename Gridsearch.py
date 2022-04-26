
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

data = pd.read_csv(r'data\heart.csv') 
x = data.drop(['output'], axis=1) # features - train and val data
y = data['output']
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)



hyper_params = {
    "n_estimators": [100, 200, 500],
    "random_state": [0, 42, 100],
    "max_depth": [3, 6, 9],
    "min_samples_split": [2, 3, 4]
}

clf = ExtraTreesClassifier()

GS = GridSearchCV(estimator=clf, param_grid=hyper_params, scoring = 'accuracy', cv=5, refit='accuracy')

GS.fit(x_train, y_train)
# print(GS.best_estimator_)
# print(GS.best_params_)
scores = GS.best_params_
print(scores)