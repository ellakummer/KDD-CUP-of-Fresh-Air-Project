import numpy as np
import pandas as pd
import random

# ----------- TEST JOINING DATA FRAMES -----------------

#df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3'],'C': ['C0', 'C1', 'C2', 'C3'],'D': ['D0', 'D1', 'D2', 'D3']},index=[0, 1, 2, 3])
#df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],'D': ['D2', 'D3', 'D6', 'D7'],'F': ['F2', 'F3', 'F6', 'F7']}, index=[2, 3, 6, 7])
s1 = pd.Series(['X0', 'X1', 'X2', 'X3'], name='X')
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3'],'C': ['C0', 'C1', 'C2', 'C3'],'D': ['D0', 'D1', 'D2', 'D3']})
df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],'D': ['D2', 'D3', 'D6', 'D7'],'F': ['F2', 'F3', 'F6', 'F7']})

result = pd.concat([df4, s1], axis=1)

print(df1)
print(df4)
print(s1)
print(result)

partial_results = result.iloc[0:2,:]
print("partial results : ")
print(partial_results)

df = pd.DataFrame([[2, 3], [5, 6], [8, 9]],
     index=['cobra', 'viper', 'sidewinder'],
     columns=['max_speed', 'shield'])

df2 = pd.DataFrame([[2, 'A'], [5, 'B'], [8, 'C']],
     columns=['max_speed', 'shield'])

print(df)
print(df.loc['viper'])

print(df2)
DD = df2.loc[df2['shield']=='A',:]
print(DD)
