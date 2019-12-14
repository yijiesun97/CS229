import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from Plot import plot_confusion_matrix, plot_roc_curve
from sklearn.preprocessing import StandardScaler

## read in data
data = pd.read_csv("Processed.csv")
x = data.loc[:, 'employee_count':'Wyoming']
y = data['in_business']

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=229, stratify=y)

## Random Forest
clf = RandomForestClassifier(n_estimators=600, min_samples_split=10, min_samples_leaf=10, max_features='sqrt',
                             max_depth=20, bootstrap=False, random_state=229, class_weight='balanced')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
train_pred = clf.predict(x_train)

## train metrics
print("accuracy:", metrics.accuracy_score(y_train, train_pred))
print("recall:", metrics.recall_score(y_train, train_pred, pos_label="Success"))
print("precision:", metrics.precision_score(y_train, train_pred, pos_label="Success"))
print("f1-score:", metrics.f1_score(y_train, train_pred, pos_label="Success"))
print("======Classification report========")
print(metrics.classification_report(y_train, train_pred))

plot_confusion_matrix(y_train, train_pred, classes=["Failure", "Success"])
plt.show()

probs = clf.predict_proba(x_train)[:, 1]
plot_roc_curve(y_train, probs)

## test metrics
print("accuracy:", metrics.accuracy_score(y_test, y_pred))
print("recall:", metrics.recall_score(y_test, y_pred, pos_label="Success"))
print("precision:", metrics.precision_score(y_test, y_pred, pos_label="Success"))
print("f1-score:", metrics.f1_score(y_test, y_pred, pos_label="Success"))
print("======Classification report========")
print(metrics.classification_report(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred, classes=["Failure", "Success"])
plt.show()

probs = clf.predict_proba(x_test)[:, 1]
plot_roc_curve(y_test, probs)

## plot feature importance
for name, importance in zip(list(data.columns)[4:], clf.feature_importances_):
    print(name, "=", importance)

features = list(data.columns)[4:]
importances = clf.feature_importances_
indices = np.argsort(importances)[-20:]

plt.title('Top 20 Most Important Features')
plt.barh(range(len(indices)), importances[indices], color='darkred', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize=8)
plt.xlabel('Relative Importance')
plt.show()