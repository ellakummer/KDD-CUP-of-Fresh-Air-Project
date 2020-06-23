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


print("----------------------------- LONDON ---------------------------------")

print("--------------------------- LOAD DATA ---------------------------------")

X_train = np.empty([200,3,8])
X_train_reduc = np.empty([200,3,3])

# X_train_app = np.empty([200,24])
X_train_app = np.empty([200,8])
X_train_app_reduc = np.empty([200,3])

y_train = np.empty([200,2])
y_train_app1 = np.empty([200])
y_train_app2 = np.empty([200])
y_train_app3 = np.empty([200])

X_val = np.empty([100,3,8])
X_val_reduc = np.empty([100,3,3])
# X_val_app = np.empty([100,24])
X_val_app = np.empty([100,8])
X_val_app_reduc = np.empty([100,3])

y_val = np.empty([100,2])
y_val_app1 = np.empty([100])
y_val_app2 = np.empty([100])
y_val_app3 = np.empty([100])

days = 2
# stations = ['BL0', 'BX9', 'CD1', 'CD9', 'CR8', 'CT2', 'CT3', 'GB0', 'GN0', 'GN3', 'GR4', 'GR9', 'HV1', 'KF1', 'LW2', 'MY7', 'RB7', 'ST5', 'TD5', 'TH4']
stations = ['BL0', 'CD1', 'CD9', 'GN0', 'GN3', 'GR4', 'GR9', 'HV1', 'KF1', 'LW2', 'MY7', 'ST5', 'TH4']
# stations = ['BL0'] # pick one
for s in stations :
    #print(s)
    all = pd.read_csv('../final_project_data/merge/'+s+'.csv')
    #print(all.shape)
    if s == 'CT2' :
        startDateTest = 9240
        endDateTest = 9287
    else :
        startDateTest = 10656
        endDateTest = 10703

    # SEE YOU LATER TESTS :)
    #X_test = all.loc[startDateTest:endDateTest,['temperature','pressure','humidity','wind_direction','wind_speed/kph','PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
    #y_test = all.loc[startDateTest:endDateTest,['PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
    # X = all[['temperature','pressure','humidity','wind_direction','wind_speed/kph','PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)', 'utc_time']].to_numpy()
    X = all[['temperature','pressure','humidity','wind_direction','wind_speed/kph','PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
    X_reduc = all[['PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
    y = all[['PM2.5 (ug-m3)','PM10 (ug-m3)']].to_numpy()
    # print('shape y : ', y.shape)


    # OPTION 1 : USE THE DATA FROM TWO DAYS BEFORE TO PREDICT
    # X_train1, X_val1 = X[:200], X[200:300]
    # y_train1, y_val1 = y[days*24:200+days*24], y[200+days*24:300+days*24]

    X_train1, X_val1 = X[startDateTest-300-days*24:startDateTest-100-days*24], X[startDateTest-100-days*24:startDateTest-days*24]
    X_train1_reduc, X_val1_reduc = X_reduc[startDateTest-300-days*24:startDateTest-100-days*24], X_reduc[startDateTest-100-days*24:startDateTest-days*24]
    y_train1, y_val1 = X[startDateTest-300:startDateTest-100], X[startDateTest-100:startDateTest]
    '''
    print('X_train1[0] : ', X_train1[0])
    print('X_train1[last] : ', X_train1[-1])
    print('X_val1[0] : ', X_val1[0])
    print('X_val1[last] : ', X_val1[-1])
    print('y_train1[0] : ', y_train1[0])
    print('y_train1[last] : ', y_train1[-1])
    print('y_val1[0] : ', y_val1[0])
    print('y_val1[last] : ', y_val1[-1])


    print("option 1 : ")
    print(X_train1.shape)
    print(y_train1.shape)
    print(X_val1.shape)
    print(y_val1.shape)
    '''

    # OPTION 2 : USE THE PREVIOUS HOUR TO PREDICT
    #print("option 2 : ")
    # X_train2, X_val2 = X[:200], X[200:300]
    # y_train2, y_val2 = y[1:201], y[201:301]

    X_train2, X_val2 = X[startDateTest-300-1:startDateTest-100-1], X[startDateTest-100-1:startDateTest-1]
    X_train2_reduc, X_val2_reduc = X_reduc[startDateTest-300-1:startDateTest-100-1], X_reduc[startDateTest-100-1:startDateTest-1]
    y_train2, y_val2 = X[startDateTest-300:startDateTest-100], X[startDateTest-100:startDateTest]
    '''
    print('X_train2[0] : ', X_train2[0])
    print('X_train2[last] : ', X_train2[-1])
    print('X_val2[0] : ', X_val2[0])
    print('X_val2[last] : ', X_val2[-1])
    print('y_train2[0] : ', y_train2[0])
    print('y_train2[last] : ', y_train2[-1])
    print('y_val2[0] : ', y_val2[0])
    print('y_val2[last] : ', y_val2[-1])

    print(X_train2.shape)
    print(y_train2.shape)
    print(X_val2.shape)
    print(y_val2.shape)
    '''
    # OPTION 3 : USE THE THREE PREVIOUS HOURS TO PREDICT (with separate vectors)
    # OR USE THE THREE PREVIOUS HOURS TO PREDICT (as one vector)
    #print("option 3 : ")
    # X_train3 = np.array([np.array([ X[i],X[i+1],X[i+2]])for i in range(200-3+1)]) #0,1,2... 1,2,3.... 197,198,199
    # X_val3 = np.array([np.array([ X[i],X[i+1],X[i+2]])for i in range(198,300-3+1)])
    # y_train3, y_val3 = y[3:201], y[201:301]
    # X_train4 = np.array([np.append(X[i],np.append(X[i+1],X[i+2]))for i in range(200-3+1)   ]) #0,1,2... 1,2,3.... 197,198,199
    # X_val4 = np.array([ np.append(X[i],np.append(X[i+1],X[i+2]))for i in range(198,300-3+1)   ])
    # y_train4,y_val4 = y[3:201,0], y[201:301,0]
    # y_train5,y_val5 = y[3:201,1], y[201:301,1]

    # X_train3 = np.array([np.array([ X[i],X[i+1],X[i+2]])for i in range(200-3+1)]) #0,1,2... 1,2,3.... 197,198,199
    # X_val3 = np.array([np.array([ X[i],X[i+1],X[i+2]])for i in range(198,300-3+1)])
    # y_train3, y_val3 = y[3:201], y[201:301]

    # X_train4 = np.array([np.append(X[i],np.append(X[i+1],X[i+2]))for i in range(200-3+1)   ]) #0,1,2... 1,2,3.... 197,198,199
    # X_val4 = np.array([ np.append(X[i],np.append(X[i+1],X[i+2]))for i in range(198,300-3+1)   ])
    # y_train4,y_val4 = y[3:201,0], y[201:301,0]
    # y_train5,y_val5 = y[3:201,1], y[201:301,1]


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


    '''
    print(X_train3[0])
    print(X_val3[0])
    print(X_train3.shape)
    print(y_train3.shape)
    print(X_val3.shape)
    print(y_val3.shape)

    print("concat : ")
    X_train = np.append(X_train, X_train3,axis=0)
    y_train = np.append(y_train, y_train3,axis=0)
    X_val = np.append(X_val, X_val3,axis=0)
    y_val = np.append(y_val, y_val3,axis=0)
    '''
    #
    # print(X_train_app.shape)
    # print(X_train4.shape)
    X_train = np.append(X_train, X_train3,axis=0)
    X_train_reduc = np.append(X_train_reduc, X_train3_reduc,axis=0)

    X_train_app = np.append(X_train_app, X_train4,axis=0)
    X_train_app_reduc = np.append(X_train_app_reduc, X_train4_reduc,axis=0)

    y_train = np.append(y_train, y_train3,axis=0)
    y_train_app1 = np.append(y_train_app1,y_train4,axis=0)
    y_train_app2 = np.append(y_train_app2,y_train5,axis=0)
    #y_train_app3 = np.append(y_train_app3,y_train6,axis=0)


    X_val = np.append(X_val, X_val3,axis=0)
    X_val_reduc = np.append(X_val_reduc, X_val3_reduc,axis=0)

    X_val_app = np.append(X_val_app, X_val4,axis=0)
    X_val_app_reduc = np.append(X_val_app_reduc, X_val4_reduc,axis=0)

    y_val = np.append(y_val, y_val3,axis=0)
    y_val_app1 = np.append(y_val_app1,y_val4,axis=0)
    y_val_app2 = np.append(y_val_app2,y_val5,axis=0)




    #y_val_app3 = np.append(y_val_app3,y_val6,axis=0)

    # x = range(len(y_train5))
    # y = y_train5
    # plt.plot(x,y)
    # plt.xlabel('utc_time')
    # plt.ylabel('2.5PM Level')
    # plt.title('2.5PM from data training : '+ s)
    # plt.show()

'''
print("final matrix : ")
X_train = X_train[198:]
y_train = y_train[198:]
X_val= X_val[100:]
y_val = y_val[100:]
'''

#print("final matrix : ")
# X_train = X_train[198:]
# X_train_app = X_train_app[198:]
#
# y_train = y_train[198:]
# y_train_app1 = y_train_app1[198:]
# y_train_app2 = y_train_app2[198:]
# #y_train_app3 = y_train_app3[198:]

X_train = X_train[200:]
X_train_reduc = X_train_reduc[200:]

X_train_app = X_train_app[200:]
X_train_app_reduc = X_train_app_reduc[200:]

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
#y_val_app3 = y_val_app3[100:]

X_train_app_total = np.append(X_train_app, X_val_app, axis=0)
X_train_app_total_reduc = np.append(X_train_app_reduc, X_val_app_reduc, axis=0)

# print(X_train_app.shape)
# print(X_val_app.shape)
#
# print(X_train_app_total.shape)
# print(X_train_app_total_reduc.shape)

y_train_app1_total = np.append(y_train_app1, y_val_app1, axis=0)
y_train_app2_total = np.append(y_train_app2, y_val_app2, axis=0)

# print('shape X_train_app : ', X_train_app.shape)
# print('shape y_train_app1 : ', y_train_app1.shape)

print("---------------------------PREDICT ---------------------------------")

'''
model = GradientBoostingRegressor(n_estimators=esti, learning_rate=lr, max_depth=depth, random_state=0, loss='ls').fit(X_train_app_total_reduc, y_train_app2_total)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train_app_total_reduc, y_train_app2_total, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# print(n_scores)
if mean(n_scores) < max_score :
    max_score = mean(n_scores)
'''
print('PM2.5')
print("Using gradient tree bosting : ")
print("with n_estimators : ", 1750)
print("with learning_rate : ", 0.5)
print("with max_depth : ", 6)
est = GradientBoostingRegressor(n_estimators=1750, learning_rate=0.5, max_depth=6, random_state=0, loss='ls').fit(X_train_app_total_reduc, y_train_app1_total)
pred = est.predict(X_train_app_total_reduc)
error = mean_squared_error(y_train_app1_total,pred)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(est, X_train_app_total_reduc, y_train_app1_total, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')

print('mean error : ', error)
print('mean error 2: ', mean(n_scores))
print('real : ', y_train_app1_total[:20])
print('pred : ', pred[:20])

print('#######################################################################')
print('PM10')
print("Using gradient tree bosting : ")
print("with n_estimators : ", 1750 )
print("with learning_rate : ", 0.5)
print("with max_depth : ", 6)
est = GradientBoostingRegressor(n_estimators=1750, learning_rate=0.5, max_depth=6, random_state=0, loss='ls').fit(X_train_app_total_reduc, y_train_app2_total)
pred = est.predict(X_train_app_total_reduc)
error = mean_squared_error(y_train_app2_total,pred)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(est, X_train_app_total_reduc, y_train_app2_total, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')

print('mean error : ', error)
print('mean error 2: ', mean(n_scores))
print('real : ', y_train_app2_total[:20])
print('pred : ', pred[:20])


'''
---------------------------------- BEIJING -------------------------------------
'''
