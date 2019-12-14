import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from Plot import plot_confusion_matrix, plot_roc_curve

## read in data
data = pd.read_csv("Processed.csv")
x = data.loc[:, 'employee_count':'Wyoming']
y = data['in_business']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=229, stratify=y)

## logistic regression
logisticRegr = LogisticRegression(penalty='none', solver='saga', tol=1e-3, max_iter=2000)
logisticRegr.fit(x_train, y_train)
y_pred = logisticRegr.predict(x_test)
train_pred = logisticRegr.predict(x_train)

## train metrics
print("accuracy:", metrics.accuracy_score(y_train, train_pred))
print("recall:", metrics.recall_score(y_train, train_pred, pos_label="Success"))
print("precision:", metrics.precision_score(y_train, train_pred, pos_label="Success"))
print("f1-score:", metrics.f1_score(y_train, train_pred, pos_label="Success"))
print("======Classification report========")
print(metrics.classification_report(y_train, train_pred))

plot_confusion_matrix(y_train, train_pred, classes=["Failure", "Success"])
plt.show()

y_score = logisticRegr.decision_function(x_train)
plot_roc_curve(y_train, y_score)

## test metrics
print("accuracy:", metrics.accuracy_score(y_test, y_pred))
print("recall:", metrics.recall_score(y_test, y_pred, pos_label="Success"))
print("precision:", metrics.precision_score(y_test, y_pred, pos_label="Success"))
print("f1-score:", metrics.f1_score(y_test, y_pred, pos_label="Success"))
print("======Classification report========")
print(metrics.classification_report(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred, classes=["Failure", "Success"])
plt.show()

y_score = logisticRegr.decision_function(x_test)
plot_roc_curve(y_test, y_score)