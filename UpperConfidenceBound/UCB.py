# import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# implement UCB
# for each round n and each ad i calculate 
## N(n) the number of times the ad i was selected up to round n
## R(n) the sum of rewards of the ad i up to round n

## average reward of ad i up to round n
# r(n) = R(n)/N(n)

# confidence interval 
# D(n) = sqrt((2*log(n))/(2N(n)))

# select the ad with max UCB = r(n) + D(n)


import math
N = 10000
d = 10

ads_selected = []
number_of_selections = [0]*d
sums_of_rewards = [0]*d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0

    for i in range(0, d):
        # check if ad is selected was atleast 1s
        if number_of_selections[i]>0:
            average_reward = sums_of_rewards[i]/ number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if (upper_bound > max_upper_bound):
            max_upper_bound =  upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad]+=1   
    reward = dataset.values[n, ad]      
    sums_of_rewards[ad] += reward
    total_reward += reward
#print(ads_selected)
# visualizing results
plt.hist(ads_selected)
plt.title('Histogram: Ads selections')    

plt.show()

from collections import Counter
cnt = Counter(ads_selected)
print(cnt)

