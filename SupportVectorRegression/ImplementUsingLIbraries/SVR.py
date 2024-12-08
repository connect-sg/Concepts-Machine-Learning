# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# create dataset

dataset = pd.read_csv('Position_Salaries.csv')
#print(dataset)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values
# print(X)

# change y to 2d array
y = y.reshape(len(y),1)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)

sc_y  =StandardScaler()
y = sc_y.fit_transform(y)

# train SVR model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))
print(y_pred)


# visualization 
plt.scatter(sc_x.inverse_transform(X),sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'blue')
plt.show()