#  import libraries
import numpy as np
import pandas as pd

# create dataset
df = pd.read_csv('california_housing_train.csv')
#print(df)


df.corr()['totalBedrooms']
# negative value indicates decareasing slop. FOr example, housingMedianAge will decrease with increase in totalBedrooms 
# Positive value indicates increasing slop. FOr example, total rooms increase with increase in total bedrooms
# Fields of instreset will be ones with strong correlation i.e. value near to 1 or -1
'''
longitude           0.071802
latitude           -0.069373
housingMedianAge   -0.320434
totalRooms          0.928403
totalBedrooms       1.000000
population          0.881169
households          0.980920
medianIncome       -0.013495
medianHouseValue    0.045783
Name: totalBedrooms, dtype: float64
'''

from copy import deepcopy
# pre-process data
bedrooms = df['totalBedrooms']
df = df.drop(['totalBedrooms', 'longitude', 'latitude', 'housingMedianAge', 'medianIncome', 'medianHouseValue'], axis = 1)
df['bedrooms'] = bedrooms
#print(df)


# convert to numpy
df_np  = df.to_numpy()
# print(df_np.shape)

X_train, y_train = df_np[:, :3], df_np[:, -1]
# print(X_train.shape, y_train.shape)

# Assuming total_bedrooms[i] = alpha + (beta_1 * population[i]) + (beta_2 * households[i]) + (beta_3 * total_rooms[i]) + error
# Generally: y[i] = alpha + (beta_1 * x_1[i]) + (beta_2 * x_2[i]) + (beta_3 * x_3[i]) + error
# Model:     y_hat[i] = alpha_hat + (beta_1_hat * x_1[i]) + (beta_2_hat * x_2[i]) + (beta_3_hat * x_3[i])


def get_predictions(model, X):

     (n ,p_minus_one) = X.shape
     p = p_minus_one + 1

     new_x = np.ones(shape = (n,p))
     new_x[:, 1:] = X
     return np.dot(new_x, model)


test_model = np.array([1,2,1,4])
get_predictions(test_model, X_train).shape
