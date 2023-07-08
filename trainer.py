import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

dataset = pd.read_csv('bothHand.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

dataset = pd.read_csv('singleHand.csv')
X1 = dataset.iloc[:, 1:].values
y1 = dataset.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.1, random_state=10)

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

joblib.dump(classifier, "svm_2h")
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print("____2h____")
print(cm)
print(accuracy_score(y_test, y_pred), "____________\n")

classifier.fit(X_train1, y_train1)
joblib.dump(classifier, "svm_1h")

y_pred = classifier.predict(X_test1)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test1, y_pred)
print("____2h____")
print(cm)
print(accuracy_score(y_test1, y_pred), "____________\n")
