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

# Initialising the sets

X_train = np.empty([200,3,8])
X_train_filtered = np.empty([200,3,3])

X_train_appended = np.empty([200,8])
X_train_appended_reduced = np.empty([200,3])

y_train = np.empty([200,2])
y_train_app1 = np.empty([200])
y_train_app2 = np.empty([200])
y_train_app3 = np.empty([200])

X_val = np.empty([100,3,8])
X_val_reduc = np.empty([100,3,3])

X_val_app = np.empty([100,8])
X_val_app_reduc = np.empty([100,3])

y_val = np.empty([100,2])
y_val_app1 = np.empty([100])
y_val_app2 = np.empty([100])
y_val_app3 = np.empty([100])

days = 2

stations = ['BL0', 'CD1', 'CD9', 'GN0', 'GN3', 'GR4', 'GR9', 'HV1', 'KF1', 'LW2', 'MY7', 'ST5', 'TH4']
for s in stations :
    all = pd.read_csv('../final_project_data/merge/'+s+'.csv')
    if s == 'CT2' :
        startDateTest = 9240
        endDateTest = 9287
    else :
        startDateTest = 10656
        endDateTest = 10703

    # X = all[['temperature','pressure','humidity','wind_direction','wind_speed/kph','PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)', 'utc_time']].to_numpy()
    X = all[['temperature','pressure','humidity','wind_direction','wind_speed/kph','PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
    X_reduc = all[['PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
    y = all[['PM2.5 (ug-m3)','PM10 (ug-m3)']].to_numpy()

    # OPTION 1 : USE THE DATA FROM TWO DAYS BEFORE TO PREDICT

    X_train1, X_val1 = X[startDateTest-300-days*24:startDateTest-100-days*24], X[startDateTest-100-days*24:startDateTest-days*24]
    X_train1_reduc, X_val1_reduc = X_reduc[startDateTest-300-days*24:startDateTest-100-days*24], X_reduc[startDateTest-100-days*24:startDateTest-days*24]
    y_train1, y_val1 = X[startDateTest-300:startDateTest-100], X[startDateTest-100:startDateTest]

    # OPTION 2 : USE THE PREVIOUS HOUR TO PREDICT

    X_train2, X_val2 = X[startDateTest-300-1:startDateTest-100-1], X[startDateTest-100-1:startDateTest-1]
    X_train2_reduc, X_val2_reduc = X_reduc[startDateTest-300-1:startDateTest-100-1], X_reduc[startDateTest-100-1:startDateTest-1]
    y_train2, y_val2 = X[startDateTest-300:startDateTest-100], X[startDateTest-100:startDateTest]

    # OPTION 3 : USE THE THREE PREVIOUS HOURS TO PREDICT (with separate vectors)
    # OR USE THE THREE PREVIOUS HOURS TO PREDICT (as one vector)

    X_train3 = np.array([np.array([ X[i],X[i+1],X[i+2]])for i in range(startDateTest-300-3,startDateTest-100-3)]) #0,1,2... 1,2,3.... 197,198,199
    X_train3_reduc = np.array([np.array([ X_reduc[i],X_reduc[i+1],X_reduc[i+2]])for i in range(startDateTest-300-3,startDateTest-100-3)]) #0,1,2... 1,2,3.... 197,198,199

    X_val3 = np.array([np.array([ X[i],X[i+1],X[i+2]])for i in range(startDateTest-100-3,startDateTest-3+1-1)])
    X_val3_reduc = np.array([np.array([ X_reduc[i],X_reduc[i+1],X_reduc[i+2]])for i in range(startDateTest-100-3,startDateTest-3+1-1)])

    y_train3, y_val3 = y[startDateTest-300:startDateTest-100], y[startDateTest-100:startDateTest]

    # USE THE PREVIOUS HOUR TO PREDICT
    X_train4 = np.array([X[i] for i in range(startDateTest-300-1,startDateTest-100-1)]) #0,1,2... 1,2,3.... 197,198,199
    X_train4_reduc = np.array([X_reduc[i] for i in range(startDateTest-300-1,startDateTest-100-1)]) #0,1,2... 1,2,3.... 197,198,199

    X_val4 = np.array([X[i] for i in range(startDateTest-100-1,startDateTest-1+1-1)])
    X_val4_reduc = np.array([X_reduc[i] for i in range(startDateTest-100-1,startDateTest-1+1-1)])

    y_train4, y_val4 = y[startDateTest-300:startDateTest-100,0], y[startDateTest-100:startDateTest,0]
    y_train5, y_val5 = y[startDateTest-300:startDateTest-100,1], y[startDateTest-100:startDateTest,1]

    # concat
    X_train = np.append(X_train, X_train3,axis=0)
    X_train_filtered = np.append(X_train_filtered, X_train3_reduc,axis=0)
    X_train_appended = np.append(X_train_appended, X_train4,axis=0)
    X_train_appended_reduced = np.append(X_train_appended_reduced, X_train4_reduc,axis=0)
    y_train = np.append(y_train, y_train3,axis=0)
    y_train_app1 = np.append(y_train_app1,y_train4,axis=0)
    y_train_app2 = np.append(y_train_app2,y_train5,axis=0)
    X_val = np.append(X_val, X_val3,axis=0)
    X_val_reduc = np.append(X_val_reduc, X_val3_reduc,axis=0)
    X_val_app = np.append(X_val_app, X_val4,axis=0)
    X_val_app_reduc = np.append(X_val_app_reduc, X_val4_reduc,axis=0)
    y_val = np.append(y_val, y_val3,axis=0)
    y_val_app1 = np.append(y_val_app1,y_val4,axis=0)
    y_val_app2 = np.append(y_val_app2,y_val5,axis=0)

# getting the final data on which to train
X_train = X_train[200:]
X_train_filtered = X_train_filtered[200:]

X_train_appended = X_train_appended[200:]
X_train_appended_reduced = X_train_appended_reduced[200:]

y_train = y_train[200:]
y_train_app1 = y_train_app1[200:]
y_train_app2 = y_train_app2[200:]

X_val= X_val[100:]
X_val_reduc = X_val_reduc[100:]

X_val_app = X_val_app[100:]
X_val_app_reduc = X_val_app_reduc[100:]

y_val = y_val[100:]
y_val_app1 = y_val_app1[100:]
y_val_app2 = y_val_app2[100:]

# concatenate training and validation test :
X_train_appended_total = np.append(X_train_appended, X_val_app, axis=0)
X_train_appended_total_reduced = np.append(X_train_appended_reduced, X_val_app_reduc, axis=0)
y_train_PM25_total = np.append(y_train_app1, y_val_app1, axis=0) # predict PM2.5
y_train_PM10_total = np.append(y_train_app2, y_val_app2, axis=0) # predict PM10
