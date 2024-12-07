###### import libraries #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


##### create dataset #####

#create dataframe
data = pd.read_csv('Data.csv')     
# print(data)

# independent variable 
X = data.iloc[:, 0:-1].values
# print(X)

# dependent variable
y = data.iloc[:,-1].values
# print(y)


##### Handle missing data #####

# create instance of SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

# apply imputer to data
imputer.fit(X[:, 1:3])

# tranform 
X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X)

##### Encoding Categorical data #####

### Encoding independent variable

# specify type of transformation to be done and encoding type

### transformer -- [('encoder', OneHotEncoder(), [0])] --> type of tranformation is encoding, 2nd specify type of encoding i.e. OneHotEncoder 
### and 3rd column in data to which transformation is to be applied

### remainder - passthrough. TO keep other columns in dataset that are not encoded. If passthrough is not used, it will only keep the encoded column

ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])] , remainder = 'passthrough')

# fit and transform the column
# convert it into numpy array
X = np.array(ct.fit_transform(X))
# print(X)


### Encoding dependent variable
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)   # [0 1 0 0 1 1 0 1 0 1]


##### splitting the dataset into training and test set #####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
# print(X_train, X_test, y_train, y_test)


#### Feature Scaling #####
sc = StandardScaler()

# only apply feature scaling on numerical values not encoded or dummy variables
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:]  = sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)