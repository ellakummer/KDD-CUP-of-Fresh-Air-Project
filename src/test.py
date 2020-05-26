import numpy as np

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
