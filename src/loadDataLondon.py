import csv
import numpy as np
import sklearn
import pandas as pd

days = 2

print("-------------------------- MERGE DATAS  --------------------------")

stations = ['BL0', 'BX1', 'BX9', 'CD1', 'CD9', 'CR8', 'CT2', 'CT3', 'GB0', 'GN0', 'GN3', 'GR4', 'GR9', 'HR1', 'HV1', 'KC1', 'KF1', 'LH0', 'LW2', 'MY7', 'RB7', 'ST5', 'TD5', 'TH4']
load_datas = []
for s in stations :
    print(s)
    all = pd.read_csv('../final_project_data/merge/'+s+'.csv')
    print(all.shape)
    X = all[['temperature','pressure','humidity','wind_direction','wind_speed/kph','PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
    y = all[['PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
    print(X.shape)
    print(y.shape)
    print(X[0]) # tests see right id station
    print(y[0])
    X_train, X_val = X[:200], X[200:300]
    y_train, y_val = y[days*24:200+days*24], y[200+days*24:300+days*24]
