from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RFR import RandomForest


data = datasets.load_breast_cancer()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1234)

def accuracy(y_test, y_pred):
    accuracy = np.sum(y_test==y_pred)/len(y_test)
    return accuracy

clf = RandomForest()
pred = clf.predict(X_test)    

print(accuracy(y_test, pred))