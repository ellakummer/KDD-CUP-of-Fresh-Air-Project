import numpy as np
import pandas as pd
import random

# ----------- TEST JOINING DATA FRAMES -----------------

#df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3'],'C': ['C0', 'C1', 'C2', 'C3'],'D': ['D0', 'D1', 'D2', 'D3']},index=[0, 1, 2, 3])
#df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],'D': ['D2', 'D3', 'D6', 'D7'],'F': ['F2', 'F3', 'F6', 'F7']}, index=[2, 3, 6, 7])
s1 = pd.Series(['X0', 'X1', 'X2', 'X3'], name='X')
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3'],'C': ['C0', 'C1', 'C2', 'C3'],'D': ['D0', 'D1', 'D2', 'D3']})
df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],'D': ['D2', 'D3', 'D6', 'D7'],'F': ['F2', 'F3', 'F6', 'F7']})
df3 = pd.DataFrame({'F': ['F2', 'F3', 'F6', 'F7'],'E': ['E2', 'E3', 'E6', 'E7'],'D': ['D1', 'D2', 'D3', 'D4']})
df5 = pd.DataFrame({'F': ['A2', 'A3', 'A6', 'A7'],'G': ['B2', 'B3', 'B6', 'B7'],'D': ['C1', 'C2', 'C3', 'C4']})
df7 = pd.DataFrame({'F': ['B2', 'B3', 'B6', 'B7'],'G': ['C2', 'C3', 'C6', 'C7'],'D': ['D1', 'D2', 'D3', 'D4']})
df6 = pd.DataFrame({'F': [],'G': [],'D': []})
df6 = pd.DataFrame({'F': [],'G': [],'D': []})




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

pi = 3.14159265359
text = float(f"{pi:.1f}")
'''
formatted_pi = float(text)
print(formatted_pi)
'''
print(text)
print(type(text))
print(type(pi))

print("test joins : ")
print(df1.columns)
print(df3.columns)
result = pd.merge(left=df1, right=df3, left_on='D', right_on='D')
print(df1)
print(df3)
print(result)
print("--")
print(df3)
print(df6)
isempty = df3.empty
print(isempty)
isempty = df6.empty
print(isempty)
print("------------")
x = "Python is "
y = "awesome"
z =  x + y
print(z)

print("------------")

a = 'hahaha'
print('lol'+a+'lol')

print("------------")
#df5, df7
result = pd.concat([df5,df7])
print(result)

print("------------")

a = float(f"{30.97:.1f}")
print(a)

print("-------")
df8 = pd.DataFrame({'F': ['F2', 'F3', 'F3', 'F7'],'E': ['E2', 'E3', 'E3', 'E7'],'D': ['D1', 'D2', 'D3', 'D4']})
print(df8)
df8.drop_duplicates(subset =['E','F'], keep = 'first', inplace = True)
print(df8)

print("-----")
data = {"Name": ["James", "Alice", "Phil", "James"],
		"Age": [24, 28, 40, 24],
		"Sex": ["Male", "Female", "Male", "Male"]}
df = pd.DataFrame(data)
print(df)
df = df.drop_duplicates()
print(df)

print("-------")

a = np.array([1,2,3])
print(a)
b = np.array([[1,2,3],[4,5,6]])
print(b)
print(np.concatenate((a, b), axis=0))
