import numpy as np
import pandas as pd


#arr = np.empty((0,3), int)
arr = np.empty((0,3))
print("Empty array:")
print(arr)
arr = np.append(arr, np.array([[10,20,30]]), axis=0)
arr = np.append(arr, np.array([[40,50,60]]), axis=0)
arr = np.append(arr, np.array([[40,50,60]]), axis=0)
arr = np.append(arr, np.array([[40,50,60]]), axis=0)
print("After adding two new arrays:")
print(arr)
print(arr.shape)

# ----------- TEST PANDAS FILLING MISSING DATAS -----------------

dff = pd.DataFrame(np.random.randn(10, 3), columns=list('ABC'))

dff.iloc[3:5, 0] = np.nan
dff.iloc[4:6, 1] = np.nan
dff.iloc[5:8, 2] = np.nan
print(dff)
# The use case of this is to fill a DataFrame with the mean of that column.
dff = dff.fillna(dff.mean())
print(dff)
