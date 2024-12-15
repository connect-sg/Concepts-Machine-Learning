# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# import dataset
dataset = pd.read_csv('Social_Network_ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# train the logistic regression model 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# predict
classifier.predict(sc.transform([[30, 87000]]))


# predict test set
y_pred = classifier.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# making confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

# accuracy score
acc = accuracy_score(y_test, y_pred)
print(acc)
