# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import datasets
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)


from sklearn.svm import svc
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


# predict 
classifier.predict(X_test)
