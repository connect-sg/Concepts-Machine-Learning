# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans

# Elbow method to find the optimal numbers of cluster
'''
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Eblow method')
plt.xlabel('number of clusters') 
plt.ylabel('WCSS')
plt.show()  
'''
# identified 5 clusters using elbow method

# train the k-means method
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_pred = kmeans.fit_predict(X) 

# visual -- create scatter plot for each cluster





