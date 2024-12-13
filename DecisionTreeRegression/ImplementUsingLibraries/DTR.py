# import libarary
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# import dataset 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values


# training Decision tree Regression model on whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)


# predict a new result
y_pred = regressor.predict([[6.5]])
print(y_pred)