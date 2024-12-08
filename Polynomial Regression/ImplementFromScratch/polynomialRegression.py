# import libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# create dataset
data = pd.read_csv('Fish.csv')
# print(data)

# apply polynomial regression when data is not linear

# replace species with number
data.replace(pd.unique(data.Species),[1,2,3,4,5,6,7], inplace = True)
# print(data)

# independent and dependent variable
data_arr = np.array(data)
X = data_arr[:, :-1]
Y = data_arr[:, -1][:,np.newaxis]

# change variable to polynomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias = False)

x_poly = poly.fit_transform(X)
print(x_poly.shape)


# initalizing the weight and bias
w = np.ones([x_poly.shape[1],1])*0
b = np.ones([1,1])*0

meu = np.std(x_poly, axis = 0)[:,np.newaxis]
mean = np.mean(x_poly, axis = 0)[:,np.newaxis]
x_stand = (x_poly - mean.T)/meu.T

# iterating through to get best fitting model
lr = 0.01
n_iter = 10000
for i in range(n_iter):
    y = np.dot(x_stand,w) + b
    grad_w = -2*(np.dot(x_stand.T, (Y-y)))/y.shape[0]
    grad_b = np.mean(-2*(Y-y))

    w-=lr*grad_w
    b-=lr*grad_b





mse = np.mean(np.square(Y-y))
print(mse)