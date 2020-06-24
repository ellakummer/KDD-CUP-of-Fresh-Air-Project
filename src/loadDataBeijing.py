import csv
import numpy as np
import pandas as pd

# for random forest
import sklearn
from sklearn.datasets import make_regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor

# for gradient boosting
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report

# for GAM
from pygam import GAM, s, f
from pygam import PoissonGAM
from pygam import LogisticGAM
from pygam import LinearGAM
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV

# for Linear Regression
from sklearn.linear_model import LinearRegression

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#for xgboost
import xgboost as xgb
from xgboost import XGBRegressor

# for NN
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# tests
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# plots
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


print("----------- LOAD DATAS ------------")

X_train = np.empty([200,3,11])
X_train_O3 = np.empty([200,3,9])
X_train_PM = np.empty([200,3,9])

X_train_app = np.empty([200,11])
X_train_app_O3 = np.empty([200,9])
X_train_app_PM = np.empty([200,9])

y_train = np.empty([200,3])
y_train_app1 = np.empty([200])
y_train_app2 = np.empty([200])
y_train_app3 = np.empty([200])

X_val = np.empty([100,3,11])
X_val_O3 = np.empty([100,3,9])
X_val_PM = np.empty([100,3,9])

X_val_app = np.empty([100,11])
X_val_app_O3 = np.empty([100,9])
X_val_app_PM = np.empty([100,9])

y_val = np.empty([100,3])
y_val_app1 = np.empty([100])
y_val_app2 = np.empty([100])
y_val_app3 = np.empty([100])


days = 2
stationsStart1 = ['zhiwuyuan','dongsi','tiantan','guanyuan','wanshouxigong','aotizhongxin','nongzhanguan','wanliu','beibuxinqu','yungang','gucheng','yizhuang','dingling','badaling','miyunshuiku','donggaocun','yongledian','yufa','liulihe','qianmen','yongdingmennei','xizhimenbei','nansanhuan','dongsihuan']
stationsStart2 = ['yanqing','shunyi','daxing','pingchang','miyun']
stationsStart3 = ['pinggu']
stationsStart4 = ['mentougou']
stationsStart5 = ['tongzhou','huairou','fengtaihuayuan']
stationsStart6 = ['fangshan']
stations = ['dongsi','tiantan','guanyuan','wanshouxigong','aotizhongxin','nongzhanguan','wanliu','beibuxinqu','fengtaihuayuan','yungang','gucheng','fangshan','daxing','yizhuang','tongzhou','shunyi','pingchang','mentougou','pinggu','huairou','miyun','yanqing','dingling','badaling','miyunshuiku','donggaocun','yongledian','yufa','liulihe','qianmen','yongdingmennei','xizhimenbei','nansanhuan','dongsihuan']

for s in stations :
    all = pd.read_csv('../final_project_data/mergeBeijing/'+s+'.csv')
    if (s in stationsStart1):
        startDateTest = 10042
        endDateTest = 10089
    elif (s in stationsStart2):
        startDateTest = 9375
        endDateTest = 9422
    elif (s in stationsStart3):
        startDateTest = 9372
        endDateTest = 9419
    elif (s in stationsStart4):
        startDateTest = 9373
        endDateTest = 9420
    elif (s in stationsStart5):
        startDateTest = 9374
        endDateTest = 9421
    elif (s in stationsStart6):
        startDateTest = 9375
        endDateTest = 9420

    X_test = all.loc[startDateTest:endDateTest-1,['temperature','pressure','humidity','wind_direction','wind_speed','PM2.5','PM10','NO2','CO','O3','SO2']].to_numpy()
    X_test_O3 = all.loc[startDateTest:endDateTest-1,['temperature','pressure','humidity','wind_direction','wind_speed','PM10','NO2','CO','O3']].to_numpy()
    X_test_PMs = all.loc[startDateTest:endDateTest-1,['humidity','wind_direction','wind_speed','PM2.5','PM10','NO2','CO','O3','SO2']].to_numpy()
    y_test = all.loc[startDateTest+1:endDateTest,['PM2.5','PM10','O3']].to_numpy()
    y_test_PM25 = y_test[:,0]
    y_test_PM10 = y_test[:,1]
    y_test_O3 = y_test[:,2]

    X = all[['temperature','pressure','humidity','wind_direction','wind_speed','PM2.5','PM10','NO2','CO','O3','SO2']].to_numpy()
    X_O3 = all[['temperature','pressure','humidity','wind_direction','wind_speed','PM10','NO2','CO','O3']].to_numpy()
    X_PM = all[['humidity','wind_direction','wind_speed','PM2.5','PM10','NO2','CO','O3','SO2']].to_numpy()
    y = all[['PM2.5','PM10','O3']].to_numpy()
    y_PM5 = y[:,0]
    y_PM10 = y[:,1]
    y_O3 = y[:,2]

    # OPTION 1 : USE THE DATA FROM TWO DAYS BEFORE TO PREDICT
    X_train1, X_val1 = X[startDateTest-300-days*24:startDateTest-100-days*24], X[startDateTest-100-days*24:startDateTest-days*24]
    X_train1_03, X_val1_03 = X_O3[startDateTest-300-days*24:startDateTest-100-days*24], X_O3[startDateTest-100-days*24:startDateTest-days*24]
    X_train1_PM, X_val1_PM = X_PM[startDateTest-300-days*24:startDateTest-100-days*24], X_O3[startDateTest-100-days*24:startDateTest-days*24]
    y_train1, y_val1 = y[startDateTest-300:startDateTest-100], y[startDateTest-100:startDateTest]

    # OPTION 2 : USE THE THREE PREVIOUS HOURS TO PREDICT (with separate vectors)
    X_train3= np.array([np.array([ X[i],X[i+1],X[i+2]])for i in range(startDateTest-300-3,startDateTest-100-3)]) #0,1,2... 1,2,3.... 197,198,199
    X_train3_O3 = np.array([np.array([ X_O3[i],X_O3[i+1],X_O3[i+2]])for i in range(startDateTest-300-3,startDateTest-100-3)]) #0,1,2... 1,2,3.... 197,198,199
    X_train3_PM = np.array([np.array([ X_PM[i],X_PM[i+1],X_PM[i+2]])for i in range(startDateTest-300-3,startDateTest-100-3)]) #0,1,2... 1,2,3.... 197,198,199
    X_val3 = np.array([np.array([ X[i],X[i+1],X[i+2]])for i in range(startDateTest-100-3,startDateTest-3+1-1)])
    X_val3_O3 = np.array([np.array([ X_O3[i],X_O3[i+1],X_O3[i+2]])for i in range(startDateTest-100-3,startDateTest-3+1-1)])
    X_val3_PM = np.array([np.array([ X_PM[i],X_PM[i+1],X_PM[i+2]])for i in range(startDateTest-100-3,startDateTest-3+1-1)])
    y_train3, y_val3 = y[startDateTest-300:startDateTest-100], y[startDateTest-100:startDateTest]

    # OPTION 3 : USE THE PREVIOUS HOUR TO PREDICT
    X_train4, X_val4 = X[startDateTest-300-1:startDateTest-100-1], X[startDateTest-100-1:startDateTest-1]
    X_train4_O3, X_val4_O3 = X_O3[startDateTest-300-1:startDateTest-100-1], X_O3[startDateTest-100-1:startDateTest-1]
    X_train4_PM, X_val4_PM = X_PM[startDateTest-300-1:startDateTest-100-1], X_PM[startDateTest-100-1:startDateTest-1]

    y_train4, y_val4 = y[startDateTest-300:startDateTest-100,0], y[startDateTest-100:startDateTest,0]
    y_train5, y_val5 = y[startDateTest-300:startDateTest-100,1], y[startDateTest-100:startDateTest,1]
    y_train6, y_val6 = y[startDateTest-300:startDateTest-100,2], y[startDateTest-100:startDateTest,2]

    # concat
    X_train = np.append(X_train, X_train3,axis=0)
    X_train_O3 = np.append(X_train_O3, X_train3_O3,axis=0)
    X_train_PM = np.append(X_train_PM, X_train3_PM,axis=0)
    X_train_app = np.append(X_train_app, X_train4,axis=0)
    X_train_app_O3 = np.append(X_train_app_O3, X_train4_O3,axis=0)
    X_train_app_PM = np.append(X_train_app_PM, X_train4_PM,axis=0)
    y_train = np.append(y_train, y_train3,axis=0)
    y_train_app1 = np.append(y_train_app1,y_train4,axis=0)
    y_train_app2 = np.append(y_train_app2,y_train5,axis=0)
    y_train_app3 = np.append(y_train_app3,y_train6,axis=0)
    X_val = np.append(X_val, X_val3,axis=0)
    X_val_O3 = np.append(X_val_O3, X_val3_O3,axis=0)
    X_val_PM = np.append(X_val_PM, X_val3_PM,axis=0)
    X_val_app = np.append(X_val_app, X_val4,axis=0)
    X_val_app_O3 = np.append(X_val_app_O3, X_val4_O3,axis=0)
    X_val_app_PM = np.append(X_val_app_PM, X_val4_PM,axis=0)
    y_val = np.append(y_val, y_val3,axis=0)
    y_val_app1 = np.append(y_val_app1,y_val4,axis=0)
    y_val_app2 = np.append(y_val_app2,y_val5,axis=0)
    y_val_app3 = np.append(y_val_app3,y_val6,axis=0)


# getting the final data on which to train
X_train = X_train[200:]
X_train_O3 = X_train_O3[200:]
X_train_PM = X_train_PM[200:]
X_train_app = X_train_app[200:]
X_train_app_O3 = X_train_app_O3[200:]
X_train_app_PM = X_train_app_PM[200:]

y_train = y_train[200:]
y_train_app1 = y_train_app1[200:]
y_train_app2 = y_train_app2[200:]
y_train_app3 = y_train_app3[200:]

X_val= X_val[100:]
X_val_O3= X_val_O3[100:]
X_val_PM= X_val_PM[100:]
X_val_app = X_val_app[100:]
X_val_app_O3 = X_val_app_O3[100:]
X_val_app_PM = X_val_app_PM[100:]

y_val = y_val[100:]
y_val_app1 = y_val_app1[100:]
y_val_app2 = y_val_app2[100:]
y_val_app3 = y_val_app3[100:]

# concatenate training and validation test :
X_train_val_PM = np.append(X_train_app_PM,X_val_app_PM,axis=0)
X_train_val_O3 = np.append(X_train_app_O3,X_val_app_O3,axis=0)
y_train_val1 = np.append(y_train_app1,y_val_app1,axis=0) # predict PM2.5
y_train_val2 = np.append(y_train_app2,y_val_app2,axis=0) # predict PM10
y_train_val3 = np.append(y_train_app3,y_val_app3,axis=0) # predict O3
