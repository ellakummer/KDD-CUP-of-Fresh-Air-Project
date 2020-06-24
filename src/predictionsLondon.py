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

print("----------- START TESTS : Linerar Regression ------------")

# Applying the best parameters to the model
X_train_l = X_train_appended_total_reduced
y_train_l1 = y_train_PM25_total
y_train_l2 = y_train_PM10_total

print('PM2.5')
reg = LinearRegression().fit(X_train_l,y_train_l1)
pred = reg.predict(X_train_l)
err = mean_squared_error(y_train_l1,pred)
print('mean error : ', err)
print('real : ', y_train_l1[:10])
print('pred : ', pred[:10])

print("----------- END TESTS : Linerar Regression ------------")

print("----------- START TESTS : Gradient Tree Boosting ------------")

# Determining the best parameters via 10 fold cross validation

print('FOR PM2.5')

n_est = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50,100,150,200,250,300,350,400,450,500,550,600,750,1000,1250,1500,1750])
learning_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06 ,0.07, 0.08, 0.09, 0.1,0.2,0.3,0.4,0.5])
max_depths = np.array([1,2,3,4,5,6,7,8,9,10])

max_score = -10000000000000000

for esti in n_est :
    for lr in learning_rates :
        for depth in max_depths :
            model = GradientBoostingRegressor(n_estimators=esti, learning_rate=lr, max_depth=depth, random_state=0, loss='ls').fit(X_train_appended_total_reduced, y_train_PM25_total)
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            n_scores = cross_val_score(model, X_train_appended_total_reduced, y_train_PM25_total, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
            print('parameters - n_estimation : ', esti, ', learning rates : ', lr, ', max depths : ', depth )
            if mean(n_scores) > max_score :
                max_score = mean(n_scores)
                best_n_est = esti
                best_lr = lr
                best_depth = depth

print("best mean squared error : ")
print(max_score)
print("with n_estimators : ")
print(best_n_est)
print("with learning_rate : ")
print(best_lr)
print("with max_depth : ")
print(best_depth)

# Applying the best parameters to the model

est = GradientBoostingRegressor(n_estimators=best_n_est, learning_rate=best_lr, max_depth=best_depth, random_state=0, loss='ls').fit(X_train_appended_total_reduced, y_train_PM25_total)
pred = est.predict(X_train_appended_total_reduced)
error = mean_squared_error(y_train_PM25_total,pred)
print('mean error : ', error)
print('real : ', y_train_PM25_total[:10])
print('pred : ', pred[:10])

print("----------- END TESTS Gradient Tree Boosting ------------")

print("----------- START TESTS : Random Forest ------------")

# Determining the best parameters via 10 fold cross validation

print('FOR PM2.5')
max_sample = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None])
max_feature = np.array([1,2])
n_estimator = np.array([10, 50, 100, 200, 300, 400, 500])
max_depth = np.array([None,1,2,3,4,5,6,7])

max_score = -100000000000000000

for max_samp in max_sample :
    for max_feat in max_feature :
        for max_dep in max_depth :
            for n in n_estimator :
                model = RandomForestRegressor(max_samples = max_samp, max_features = max_feat, n_estimators = n, max_depth = max_dep)
                cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
                n_scores = cross_val_score(model, X_train_appended_total_reduced, y_train_PM25_total, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
                if mean(n_scores) > max_score :
                    max_score = mean(n_scores)
                    best_n_est = n
                    best_max_sample = max_samp
                    best_max_feature = max_feat
                    best_max_depth = max_dep

print('Best mean error : ', max_score)
print("best n_estimator : ")
print(best_n_est)
print("best max_feature : ")
print(best_max_feature)
print("best max_sample : ")
print(best_max_sample)
print("best max_depth : ")
print(best_max_depth)

# Applying the best parameters to the model

model = RandomForestRegressor(max_samples = best_max_sample, max_features = best_max_feature, n_estimators = best_n_est, max_depth = best_max_depth)
model.fit(X_train_appended_total_reduced, y_train_PM25_total)
pred = model.predict(X_train_appended_total_reduced)
error = mean_squared_error(y_train_PM25_total,pred)
print('mean error : ', error)
print('predicted :', pred[:10])
print('real : ', y_train_PM25_total[:10])

print("----------- END TESTS Random Forest ------------")

print("----------- START TEST : GAM (generalized additive model) ------------")

# Applying the best parameters to the model
print('PM2.5')
est = LinearGAM(n_splines=10).gridsearch(X_train_appended_total_reduced, y_train_PM25_total)
pred = est.predict(X_train_appended_total_reduced)
error = mean_squared_error(y_train_PM25_total,pred)
print('mean error : ', error)
print('real : ', y_train_PM25_total[:10])
print('pred : ', pred[:10])

print("----------- END TESTS GAM ------------")

print("----------- START TESTS Multiclass Neural Network  ------------")

# Determining the best parameters via 10 fold cross validation

print('FOR PM2.5')

max_score = -1000000000000

max_iter =  [800, 850, 900, 950, 1000, 1050, 1100, 1150 ]
activation = ['identity', 'logistic', 'tanh', 'relu']

for it in max_iter:
    for acti in activation:
        model = MLPRegressor(max_iter = it, activation = acti).fit(X_train_appended_total_reduced, y_train_PM25_total)
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        n_scores = cross_val_score(model, X_train_appended_total_reduced, y_train_PM25_total, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
        if mean(n_scores) > max_score :
            max_score = mean(n_scores)
            best_it = it
            best_acti = acti

print('Best min error : ', max_score)
print('best it : ', best_it)
print('best acti : ', best_acti)

# Applying the best parameters to the model

model = MLPRegressor(max_iter = best_it, activation = best_acti).fit(X_train_appended_total_reduced, y_train_PM25_total)
pred = model.predict(X_train_appended_total_reduced)
error = mean_squared_error(y_train_PM25_total,pred)
print('mean error : ', error)
print('predicted :', pred[:10])
print('real : ', y_train_PM25_total[:10])

print("----------- END TESTS Multiclass Neural Network  ------------")

print("----------- START TESTS xgboost  ------------")

# Determining the best parameters via 10 fold cross validation

data_dmatrix = xgb.DMatrix(data=X_train_appended_total_reduced,label=abs(y_train_PM10_total/1000))
min_error = 100000
objective = np.array(["binary:logistic"])
max_depths = np.array([1,2,3,4,5,6,7,8,9,10])
colsample_bytree = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
learning_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06 ,0.07, 0.08, 0.09, 0.1,0.2,0.3,0.4,0.5])
subsample = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

for obj in objective:
    for max_dep in max_depths:
        for cb in colsample_bytree:
            for lr in learning_rates:
                for sub in subsample:
                    params = {'objective': obj,'colsample_bytree': cb,'learning_rate': lr,'max_depth': max_dep,'subsample': sub,'alpha': 10}
                    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10, num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
                    print((cv_results["test-rmse-mean"]).tail(1))
                    error = (cv_results["test-rmse-mean"]).tail(1).values[0]
                    if error < min_error :
                        min_error = error
                        best_objective = obj
                        best_depth = max_dep
                        best_colysample_bytree = cb
                        best_lr = lr
                        best_sub = sub


print("best mean squared error : ")
print(min_error)
print("with learning_rate : ")
print(best_lr)
print("with max_depth : ")
print(best_depth)
print("with objective : ")
print(best_objective)
print("with best_colysample_bytree : ")
print(best_colysample_bytree)
print("with subsample : ")
print(best_sub)

# Applying the best parameters to the model

xg_reg = xgb.XGBRegressor(objective= best_objective, colsample_bytree= best_colysample_bytree, learning_rate= best_lr, max_depth=best_depth, subsample= best_sub,alpha= 10) # FOR y_PM10
xg_reg.fit(X_train_appended_total_reduced,abs(y_train_PM10_total/1000))
pred = xg_reg.predict(X_train_appended_total_reduced)
pred = pred*1000
rmse = np.sqrt(mean_squared_error(abs(y_train_PM10_total), pred))
print("mse : ", rmse)
for i in range(10) :
    print("test : ", abs(y_train_PM10_total)[i])
    print("pred = ", pred[i])

print("----------- END TESTS xgboost  ------------")
