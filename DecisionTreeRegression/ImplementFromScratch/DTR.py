#### Credits - https://github.com/Suji04/ML_from_Scratch/blob/master/decision%20tree%20regression.ipynb #####

import pandas as pd 
import numpy as np

# import dataset
dataset = pd.read_csv('AirfoilSelfNoise.csv')
print(dataset)


# Decision tree consist of Nodes
class Node:
    def __init__(self, feature_index= None, threshold = None, left = None, right = None, var_red= None, value= None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left 
        self.right = right
        self.var_red = var_red
        self.value = value

# Tree class
class DecisionTreeRegressor:
     def __init__(self, min_samples_split = 2, max_depth = 2):
        self.root = None
        self.min_sample_split = min_samples_split
        self.max_depth = max_depth

     def  build_tree(self, dataset, curr_depth = 0):

        # split dataset in dependent and independent variable
        X, y = dataset[:, :-1], dataset[:,-1]


        num_samples, num_features = np.shape(X)
        #print(num_samples, num_features)
        
        # variable dict to hold best_split values
        best_split = {}
        
        if num_samples >= self.min_sample_split and curr_depth <= self.max_depth:
                   best_split = self.get_best_split(dataset, num_samples, num_features)
                   print( best_split.get('var_red', 'default_value') )
                   # check if variance reduction is postive
                   if best_split["var_red"]>0:
                       left_subtree = self.build_tree(best_split['dataset_left'], curr_depth+1)
                       right_subtree = self.build_tree(best_split['dataset_right'], curr_depth+1)

                       return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree, best_split['var_red'])

        leaf_value = self.calculate_leaf_value(Y)
        return Node(value = leaf_value)


     def get_best_split(self, dataset, num_samples, num_features):

        best_split = {}

        max_var_red = -float('inf')

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]

            possible_threshold = np.unique(feature_index)

            for threshold in possible_threshold:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)

                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y  = dataset[:,-1], dataset_left[:,-1], dataset_right[:, -1] 
                    curr_var_red = self.variance_reduction(y, left_y, right_y)

                    if curr_var_red > max_var_red:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left']  = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['var_red'] = curr_var_red
                        max_var_red = curr_var_red
        return best_split



     def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])  
        dataset_right = np.array([row for row in dataset if row[feature_index]>=threshold])   

        return dataset_left, dataset_right     

     def variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child)/len(parent)
        weight_r = len(r_child)/len(parent)

        reduction = np.var(parent) - (weight_l*np.var(l_child) + weight_r* np.var(r_child))

        return reduction

     def calculate_leaf_value(self, Y):
        val = np.mean(Y)
        return val

     def print_tree(self, tree = None, indent = " "):
        if not tree:
            tree = slef.root
        else:
            print("X_"+str(tree.feature_index), '<=', tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)                   


     def fit(self, X, y):
        dataset = np.concatenate((X,y), axis = 1)
        self.root = self.build_tree(dataset)


     def make_prediction(self, x, tree):
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_predicttion(x, tree.left)
        else:
             return self.make_predicttion(x, tree.right)   

     def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]  


if __name__ == '__main__':
        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values.reshape(-1,1)
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)   

        ## fit model
        regressor = DecisionTreeRegressor(min_samples_split=3, max_depth=3)
        regressor.fit(X_train,Y_train)
        regressor.print_tree()        