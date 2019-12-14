import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
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

# grid search
def grid_search(Cs):
    aucs = np.zeros(len(Cs))
    coefs = []

    for i in range(len(Cs)):
        clf = LogisticRegressionCV(Cs=[Cs[i]], cv=10, solver='liblinear', penalty='l1', max_iter=1000, tol=1e-3,
                                   scoring='roc_auc', class_weight='balanced', random_state=229).fit(x_train, y_train)
        aucs[i] = clf.score(x_train, y_train)
        coefs.append(clf.coef_.reshape(-1))
    print(aucs)
    return coefs

Cs_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4]
grid_search(Cs_grid)

Cs_fine = np.linspace(1e-2, 1, 10)
grid_search(Cs_fine)

C_best = Cs_fine[1]
print("Optimal lambda:", 1/C_best)

# logistic regression
logisticRegr = LogisticRegression(penalty='l1', C=C_best, solver='liblinear', class_weight="balanced", tol=1e-4,
                                  max_iter=2000, random_state=229)
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

## print coefficients
features = list(data.columns)[4:]
coefs = logisticRegr.coef_.reshape(-1)
indices = np.argsort(abs(coefs))

for i in range(len(indices)):
    index = int(indices[i])
    print(features[index], "=", coefs[index])

