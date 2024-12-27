# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# dependent and independent dataset
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:,-1].values

# encoding dependent varible field
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# onehotencoding on country field
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

# train test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Build ANN

# initialize ANN
ann = tf.keras.models.Sequential()

# Add input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
# second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
# output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

# compile ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# training ANN
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# making prediction and eval model
y_pred = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
print(y_pred)

# predict test results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
ac = accuracy_score(y_test, y_pred)
print(ac)