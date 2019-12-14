import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import StandardScaler
from Plot import plot_confusion_matrix, plot_roc_curve

## read in data
data = pd.read_csv("Processed.csv")
x = data.loc[:, 'employee_count':'Wyoming']
y = data['in_business']

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=229, stratify=y)

## grid search
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'laplacian']
aucs = np.zeros((5, 4, 5))

count = 1
for i in range(len(Cs)):
    for j in range(len(gammas)):
        for k in range(len(kernels)):
            print(count)
            clf = LinearSVC(penalty='l2', C=Cs[i], class_weight='balanced', random_state=229, max_iter=3000, tol=1e-3)
            feature_map = Nystroem(kernel=kernels[k], gamma=gammas[j], random_state=229, n_components=300)
            train_transformed = feature_map.fit_transform(x_train)
            aucs[i, j, k] = np.mean(cross_val_score(clf, train_transformed, y_train, scoring='roc_auc', cv=5))
            count += 1

print(aucs)

C_best = Cs[9]
print("Optimal lambda:", 1/C_best)

gamma_best = gammas[2]
print("Optimal gamma:", gamma_best)

clf = LinearSVC(penalty='l2', C=C_best, class_weight='balanced', random_state=229, max_iter=5000, tol=1e-3)
feature_map = Nystroem(kernel='laplacian', gamma=gamma_best, random_state=229, n_components=300)
train_transformed = feature_map.fit_transform(x_train)
test_transformed = feature_map.transform(x_test)
clf.fit(train_transformed, y_train)
y_pred = clf.predict(test_transformed)
train_pred = clf.predict(train_transformed)

## train metrics
print("accuracy:", metrics.accuracy_score(y_train, train_pred))
print("recall:", metrics.recall_score(y_train, train_pred, pos_label="Success"))
print("precision:", metrics.precision_score(y_train, train_pred, pos_label="Success"))
print("f1-score:", metrics.f1_score(y_train, train_pred, pos_label="Success"))
print("======Classification report========")
print(metrics.classification_report(y_train, train_pred))

plot_confusion_matrix(y_train, train_pred, classes=["Failure", "Success"])
plt.show()

y_score = clf.decision_function(train_transformed)
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

y_score = clf.decision_function(test_transformed)
plot_roc_curve(y_test, y_score)