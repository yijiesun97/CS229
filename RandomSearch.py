import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

## read in data
data = pd.read_csv("Processed.csv")
x = data.loc[:, 'employee_count':'Wyoming']
y = data['in_business']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=229, stratify=y)

## set parameter values
n_estimators = [int(x) for x in np.linspace(100, 1000, 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 20, 5)]
min_samples_split = [5, 10, 25]
min_samples_leaf = [2, 5, 10]
bootstrap = [True, False]

## create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'class_weight': ['balanced']}
pprint(random_grid)

## search for the best hyperparameters
clf = RandomForestClassifier()
clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, scoring='roc_auc',
                                cv=5, random_state=229, n_jobs=-1, return_train_score=True)
clf_random.fit(x_train, y_train)

print(clf_random.best_params_)