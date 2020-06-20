import csv
import numpy as np
import sklearn
import pandas as pd

days = 2

print("-------------------------- MERGE DATAS  --------------------------")

stations = ['dongsi','tiantan','guanyuan','wanshouxigong','aotizhongxin','nongzhanguan','wanliu','beibuxinqu','zhiwuyuan','fengtaihuayuan','yungang','gucheng','fangshan','daxing','yizhuang','tongzhou','shunyi','pingchang','mentougou','pinggu','huairou','miyun','yanqing','dingling','badaling','miyunshuiku','donggaocun','yongledian','yufa','liulihe','qianmen','yongdingmennei','xizhimenbei','nansanhuan','dongsihuan']

stationsStart1 = ['zhiwuyuan','dongsi','tiantan','guanyuan','wanshouxigong','aotizhongxin','nongzhanguan','wanliu','beibuxinqu','yungang','gucheng','yizhuang','dingling','badaling','miyunshuiku','donggaocun','yongledian','yufa','liulihe','qianmen','yongdingmennei','xizhimenbei','nansanhuan','dongsihuan']
stationsStart2 = ['yanqing','shunyi','daxing','pingchang','miyun']
stationsStart3 = ['pinggu']
stationsStart4 = ['mentougou']
stationsStart5 = ['tongzhou','huairou','fengtaihuayuan']
stationsStart6 = ['fangshan']

for s in stations :
    print(s)
    all = pd.read_csv('../final_project_data/mergeBeijing/'+s+'.csv')
    #print(all.shape)
    if (s in stationsStart1):
        startDate = 10042
        endDate = 10089
    elif (s in stationsStart2):
        startDate = 9375
        endDate = 9422
    elif (s in stationsStart3):
        startDate = 9372
        endDate = 9419
    elif (s in stationsStart4):
        startDate = 9373
        endDate = 9420
    elif (s in stationsStart5):
        startDate = 9374
        endDate = 9421
    elif (s in stationsStart6):
        startDate = 9375
        endDate = 9420

    # SI ON VEUT CONCAT : SEPARER SELECTION ET NUMPY
    X_test = all.loc[startDate:endDate,['temperature','pressure','humidity','wind_direction','wind_speed','PM2.5','PM10','NO2','CO','O3','SO2']].to_numpy()
    '''
    print(X_test[0])
    print(X_test[-1])
    '''
    y_test = all.loc[startDate:endDate,['PM2.5','PM10','O3']].to_numpy()
    X = all[['temperature','pressure','humidity','wind_direction','wind_speed','PM2.5','PM10','NO2','CO','O3','SO2']].to_numpy()
    y = all[['PM2.5','PM10','O3']].to_numpy()
    '''
    print(X.shape)
    print(y.shape)
    print(X[0]) # tests see right id station
    print(y[0])
    '''
    # OU JUTSE DECALER DE 1 (<-> UNE HEURE)... ?!
    X_train, X_val = X[:200], X[200:300]
    y_train, y_val = y[days*24:200+days*24], y[200+days*24:300+days*24]
