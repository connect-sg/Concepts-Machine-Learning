from sklearn.metrics import r2_score
r2_score(y_test, y_pred)



# additional accuracy score 

from sklearn.metrics import accuracy_score

# Assuming you have your true labels (y_true) and predicted labels (y_pred)
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 0, 1, 0, 1]

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)