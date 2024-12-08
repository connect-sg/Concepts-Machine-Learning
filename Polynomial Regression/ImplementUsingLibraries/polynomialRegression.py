#import libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

#create data
data = pd.read_csv('Position_salaries.csv')
# print(data)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# training polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)


# visualize Polynomial regression