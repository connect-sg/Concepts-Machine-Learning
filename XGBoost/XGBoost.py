# import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values
y = [0 if y==2 else 1 for y in y ]

# split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# training XGBoost
from xgboost import XGBClassifier  # XGBRegressor for regression model
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc  = accuracy_score(y_test, y_pred)
print(cm, acc)