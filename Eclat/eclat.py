# import libraries 
import pandas as pd
import numpy as np


dataset = pd.read_csv('Market_Basket_Optimisation.csv', header  = None)
print(dataset.iloc[:5, :])
transaction = []
for i in range(0, 7501):
    transaction.append([str(dataset.values[i,j]) for j in range(0,20)])
print(transaction[0])    




from apyori import apriori
rules = apriori(transactions = transaction, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)


results = list(rules)
print(results)


def inspect(results):
    lhs = [tuple(result[2][0][0])[0]for result in results]
    rhs = [tuple(result[2][0][1])[0]for result in results]
    support = [result[1] for result in results]
   
    return list(zip(lhs, rhs, support))
resultDF = pd.DataFrame(inspect(results), columns = ['LHS', 'RHS', 'Support'])

print(resultDF)
print(resultDF.nlargest(n = 10, columns = 'Support'))
