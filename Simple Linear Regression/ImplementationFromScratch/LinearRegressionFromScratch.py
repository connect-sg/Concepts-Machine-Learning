import numpy as np

class LinearRegression:
    
    def __init__(self, lr = 0.01, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initially start with wieght and bias as 0
        self.weights =  np.zeros(n_features)  
        self.bias = 0
        for _ in range(self.n_iters):
                 # predict result using y = wx+b
                 y_pred =   np.dot(X, self.weights) + self.bias
                 # derivative
                 # dw i.e. df/dw
                 dw = (1/n_samples) * np.dot(X.T, (y_pred - y))

                 # db i.e. df/db
                 db = (1/n_samples) * np.sum(y_pred - y)

                 # calculate error
                 # w = w -lr * dw
                 self.weights = self.weights - self.lr * dw
                 # b = b - lr*db
                 self.bias = self.bias - self.lr * db


    def predict(self, X):
            y_pred = np.dot(X, self.weights) + self.bias
            return y_pred         