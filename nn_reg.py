import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from Plot import plot_confusion_matrix, plot_roc_curve
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.neural_network import MLPClassifier

## read in data
data = pd.read_csv("Processed.csv")
x = data.loc[:, 'employee_count':'Wyoming']
y = data['in_business'].apply(lambda y: 1 if y == "Success" else 0)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=229, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.25, random_state=229, stratify=y_train)

## SMOTE oversampling
smote = SMOTE()
x_sm, y_sm = smote.fit_resample(x_train, y_train)
print(sorted(Counter(y_sm).items()))

## create a parameter grid
activations = ['logistic', 'tanh', 'relu']
learning_rates_init = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1]
## the optimal hidden layer size to prevenet overfitting is N_s/(alpha * (N_i + N_o)), where alpha in 2-10
hidden_sizes = [(100, 50, 20), (100, 50, 50), (100, 50, 50, 20), (100, 50, 50, 50), (100, 50, 50, 20, 20)]
alphas = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
aucs = np.zeros((3, 6, 5, 6))

## grid search
count = 1
for i in range(len(activations)):
    for j in range(len(learning_rates_init)):
        for k in range(len(hidden_sizes)):
            for l in range(len(alphas)):
                print(count)

                clf = MLPClassifier(activation=activations[i], hidden_layer_sizes=hidden_sizes[k], random_state=229,
                                    learning_rate_init=learning_rates_init[j], max_iter=500, alpha=alphas[l])
                clf.fit(x_sm, y_sm)
                fpr2, tpr2, threshold = metrics.roc_curve(y_val, clf.predict_proba(x_val)[:, 1])
                aucs[i, j, k, l] = metrics.auc(fpr2, tpr2)

                count += 1

print(aucs)

## MLP
clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(100, 50, 50, 20), random_state=229,
                                    learning_rate_init=0.0001, max_iter=500, alpha=0.05)
clf.fit(x_sm, y_sm)
y_pred = clf.predict(x_test)
train_pred = clf.predict(x_sm)

## train metrics
print("accuracy:", metrics.accuracy_score(y_sm, train_pred))
print("recall:", metrics.recall_score(y_sm, train_pred, pos_label=1))
print("precision:", metrics.precision_score(y_sm, train_pred, pos_label=1))
print("f1-score:", metrics.f1_score(y_sm, train_pred, pos_label=1))
print("======Classification report========")
print(metrics.classification_report(y_sm, train_pred))

plot_confusion_matrix(y_sm, train_pred, classes=[0, 1])
plt.show()

probs = clf.predict_proba(x_sm)[:, 1]
plot_roc_curve(y_sm, probs, pos_label=1)

## test metrics
print("accuracy:", metrics.accuracy_score(y_test, y_pred))
print("recall:", metrics.recall_score(y_test, y_pred, pos_label=1))
print("precision:", metrics.precision_score(y_test, y_pred, pos_label=1))
print("f1-score:", metrics.f1_score(y_test, y_pred, pos_label=1))
print("======Classification report========")
print(metrics.classification_report(y_test, y_pred))

plot_confusion_matrix(y_test, y_pred, classes=[0, 1])
plt.show()

probs = clf.predict_proba(x_test)[:, 1]
plot_roc_curve(y_test, probs, pos_label=1)