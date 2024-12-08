##### Simple Linear Regression model on salary_date #####

## import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

## create dataset
salary_data = pd.read_csv('data/Salary_Data.csv')
# print(salary_data)

########## Data Pre-processing ##########

## indepndent and dependent varialbes
# years of experience will be independent variablbe
# salary will be dependent variable

X = salary_data.iloc[:, 0:1].values
y = salary_data.iloc[:, 1].values
# print(X)
# print(y)


## split dataset into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1234)
#print(X_train) 
#print(X_test)


## training simple linear regression model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
# fit model
regressor.fit(X_train, y_train)

# predict
y_pred = regressor.predict(X_test)

# visualize the traiing and test set results with prediction on train set
'''
y_pred_line = regressor.predict(X_train)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize = (8,6))
m1 = plt.scatter(X_train, y_train, color = cmap(0.4), s = 10)
m2 = plt.scatter(X_test, y_test, color = cmap(0.7), s = 10)
plt.plot(X_train, y_pred_line, color = 'black', linewidth = 2, label = 'Prediction')
plt.title('Salary prediction based on year of experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
'''

# visualize the traiing and test set results with prediction on test set
y_pred_line = regressor.predict(X_test)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize = (8,6))
m1 = plt.scatter(X_train, y_train, color = cmap(0.4), s = 10)
m2 = plt.scatter(X_test, y_test, color = cmap(0.7), s = 10)
plt.plot(X_test, y_pred_line, color = 'black', linewidth = 2, label = 'Prediction')
plt.title('Salary prediction(test set) based on year of experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()












