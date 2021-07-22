# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [10, 6]
%matplotlib inline


# %%
test_result_OK = pd.read_csv('result/Setting_7_score_OK.csv')
test_result_OK_unseen = pd.read_csv('result/Setting_7_score_OK_unseen.csv')
test_result_NG = pd.read_csv('result/Setting_7_score_NG.csv')
test_result_OK_unseen = test_result_OK_unseen.loc[test_result_OK_unseen['1'] < 100, :].reset_index(drop = True)

# %%
test_result_OK_unseen = test_result_OK_unseen.loc[test_result_OK_unseen['1'] > 100, :].reset_index(drop = True)
test_result_OK_unseen

# %%
axis = '0'
fig, ax = plt.subplots(figsize=(15,10))
ax.boxplot([test_result_OK[axis], test_result_OK_unseen[axis], 
            test_result_NG[axis]], sym="b*")
plt.title('Comp')
plt.xticks([1,2,3], 
           ['OK', 'OK_unseen', 'NG'])
plt.show()

# %%

# %%
axis = '1'
threshold = 60
print("{}/{}".format(sum(test_result_NG[axis] > threshold), len(test_result_NG[axis])))
print(sum(test_result_OK[axis]>threshold))
print(sum(test_result_OK_unseen[axis]>threshold))

# %%
test_result_NG.loc[test_result_NG[axis] > threshold, :]
# %%
test_result_OK.head(5)
# %%
