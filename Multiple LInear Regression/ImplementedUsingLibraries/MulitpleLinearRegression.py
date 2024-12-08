# import libraries

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# create dataframe

df = pd.read_csv('50_startups.csv')
print(df)

# Data analysis

# check for missing data
print(df.isnull().sum())
print(df.isna().sum())

X = df.iloc[:, :-1].values
y = df.iloc[:,-1].values

# train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.20 , random_state = 1234)

# encoding the idpendent variable 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.20 , random_state = 1234)


# training the multiple Linear regression

# dummmy variable trap will be handled by library used for model 
# backward elimination will also be done by library i.e. feature selection 


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict the test results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

