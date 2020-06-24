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
from scipy import sparse
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import GridSearchCV
# for SVM with cross validation
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import utils

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
# for Multiclass Neural Network
'''
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import seaborn as sb
'''
# for Linear Regression
from sklearn.linear_model import LinearRegression
# tests
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split
# plots
import matplotlib.pyplot as plt



print("----------------------------- LONDON ---------------------------------")

print("---------------------------PREDICT ---------------------------------")

stations = ['BL0', 'CD1', 'CD9', 'GN0', 'GN3', 'GR4', 'GR9', 'HV1', 'KF1', 'LW2', 'MY7', 'ST5', 'TH4']
# stations = ['BL0'] # pick one
for s in stations :
    #print(s)
    all = pd.read_csv('../final_project_data/merge/'+s+'.csv')

    startDateTest = 10656
    endDateTest = 10703

    # SEE YOU LATER TESTS :)
    X_test = all.loc[startDateTest:endDateTest-1,['temperature','pressure','humidity','wind_direction','wind_speed/kph','PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
    y_test = all.loc[startDateTest+1:endDateTest,['PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
    y_test_PM25 = y_test[:,0]
    y_test_PM10 = y_test[:,1]

    print('-----------------------------------------------------------------------')
    print('PM2.5 for ', s)
    print("Using gradient tree bosting : ")
    est = GradientBoostingRegressor(n_estimators=800, learning_rate=0.1, max_depth=3, random_state=0, loss='ls').fit(X_test, y_test_PM25)
    pred_PM25 = est.predict(X_test)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(est, X_test, y_test_PM25, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')

    print('mean error : ', mean(n_scores))
    #print('real : ', y_test_PM25[:20])
    #print('pred : ', pred[:20])

    print('-----------------------------------------------------------------------')
    print('PM10 for ', s)
    print("Using gradient tree bosting : ")
    est = GradientBoostingRegressor(n_estimators=1750, learning_rate=0.01, max_depth=1, random_state=0, loss='ls').fit(X_test, y_test_PM25)
    pred_PM10 = est.predict(X_test)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(est, X_test, y_test_PM25, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')

    print('mean error : ', mean(n_scores))
    #print('real : ', y_test_PM25[:20])
    #print('pred : ', pred[:20])

    x = range(len(pred_PM25))
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('PM2.5 for station ' + s)
    ax2.set_xlabel('utc_time')
    ax1.set_ylabel('predicted PM2.5 Level')
    ax2.set_ylabel('real values PM2.5 Level')
    ax1.plot(x, pred_PM25)
    ax2.plot(x, y_test_PM25)
    plt.show()
    x = range(len(pred_PM10))
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('PM10 for station ' + s)
    ax2.set_xlabel('utc_time')
    ax1.set_ylabel('predicted PM10 Level')
    ax2.set_ylabel('real values PM10 Level')
    ax1.plot(x, pred_PM10)
    ax2.plot(x, y_test_PM10)
    plt.show()


'''
---------------------------------- BEIJING -------------------------------------
'''
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

    print('-----------------------------------------------------------------------')
    print('PM2.5 for ', s)
    print("Using gradient tree bosting : ")
    est = GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=3, random_state=0, loss='ls').fit(X_test_PMs, y_test_PM25)
    pred_PM25 = est.predict(X_test_PMs)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(est, X_test_PMs, y_test_PM25, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')

    print('mean error : ', mean(n_scores))

    print('-----------------------------------------------------------------------')
    print('PM10 for ', s)
    print("Using gradient tree bosting : ")
    est = GradientBoostingRegressor(n_estimators=250, learning_rate=0.07, max_depth=3, random_state=0, loss='ls').fit(X_test_PMs, y_test_PM10)
    pred_PM10 = est.predict(X_test_PMs)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(est, X_test_PMs, y_test_PM10, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')

    print('mean error : ', mean(n_scores))

    print('-----------------------------------------------------------------------')
    print('O3 for ', s)
    print("Using gradient tree bosting : ")
    est = GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=3, random_state=0, loss='ls').fit(X_test_O3, y_test_O3)
    pred_O3 = est.predict(X_test_O3)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(est, X_test_O3, y_test_O3, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')

    print('mean error : ', mean(n_scores))

    x = range(len(pred_PM25))
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('PM2.5 for station ' + s)
    ax2.set_xlabel('utc_time')
    ax1.set_ylabel('predicted PM2.5 Level')
    ax2.set_ylabel('real values PM2.5 Level')
    ax1.plot(x, pred_PM25)
    ax2.plot(x, y_test_PM25)
    plt.show()
    x = range(len(pred_PM10))
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('PM10 for station ' + s)
    ax2.set_xlabel('utc_time')
    ax1.set_ylabel('predicted PM10 Level')
    ax2.set_ylabel('real values PM10 Level')
    ax1.plot(x, pred_PM10)
    ax2.plot(x, y_test_PM10)
    plt.show()
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('O3 for station ' + s)
    ax2.set_xlabel('utc_time')
    ax1.set_ylabel('predicted O3 Level')
    ax2.set_ylabel('real values O3 Level')
    ax1.plot(x, pred_O3)
    ax2.plot(x, y_test_O3)
    plt.show()
