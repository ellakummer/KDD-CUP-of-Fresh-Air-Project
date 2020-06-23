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
# for GAM (pip3 install pygam)
from pygam import GAM, s, f
from pygam import PoissonGAM
from pygam import LogisticGAM
from pygam import LinearGAM
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
#for xgboost
import xgboost as xgb
from xgboost import XGBRegressor
# for Linear Regression
from sklearn.linear_model import LinearRegression
# tests
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split
# plots
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# https://www.hindawi.com/journals/jece/2017/5106045/

# https://medium.com/mongolian-data-stories/ulaanbaatar-air-pollution-part-1-35e17c83f70b (ici plus discret mdr)
# https://medium.com/mongolian-data-stories/part-3-the-model-b2fb9a25a07c
# https://github.com/robertritz/Ulaanbaatar-PM2.5-Prediction/blob/master/Classification%20Model/Ulaanbaatar%20PM2.5%20Prediction%20-%20Classification.ipynb

# transform into array London_historical_aqi_forecast_stations_20180331

# If the wind speed is less than 0.5m/s (nearly no wind), the value of the wind direction is 999017.
# -> put to 0

# TO PUT NAN TO AVERAGE COLUMN :
# https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns

# ALSO :
# see how to VISUALIZE the datas
# e.g. plot PMs per month ( overall visualisation, avoir une idée du truc - > dans intro rapport ? )
# e.g. afficher PMs en fonction de chaque autre donnée qu'on a (un plot par donnée externe)

# MODELS :
# GradientBoostingRegressor, svm.SVC, RandomForestRegressor, LogisticGAM
# AND TRY LINEAR REGRESSION TOO

print("----------- LOAD DATAS ------------")

X_train = np.empty([200,3,11])
X_train_O3 = np.empty([200,3,9])
X_train_PM = np.empty([200,3,9])
#X_train_app = np.empty([198,33])
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
#X_val_app = np.empty([100,33])
X_val_app = np.empty([100,11])
X_val_app_O3 = np.empty([100,9])
X_val_app_PM = np.empty([100,9])

y_val = np.empty([100,3])
y_val_app1 = np.empty([100])
y_val_app2 = np.empty([100])
y_val_app3 = np.empty([100])


days = 2
stations = ['dongsi','tiantan','guanyuan','wanshouxigong','aotizhongxin','nongzhanguan','wanliu','beibuxinqu','fengtaihuayuan','yungang','gucheng','fangshan','daxing','yizhuang','tongzhou','shunyi','pingchang','mentougou','pinggu','huairou','miyun','yanqing','dingling','badaling','miyunshuiku','donggaocun','yongledian','yufa','liulihe','qianmen','yongdingmennei','xizhimenbei','nansanhuan','dongsihuan']
# ,'zhiwuyuan'
stationsStart1 = ['zhiwuyuan','dongsi','tiantan','guanyuan','wanshouxigong','aotizhongxin','nongzhanguan','wanliu','beibuxinqu','yungang','gucheng','yizhuang','dingling','badaling','miyunshuiku','donggaocun','yongledian','yufa','liulihe','qianmen','yongdingmennei','xizhimenbei','nansanhuan','dongsihuan']
stationsStart2 = ['yanqing','shunyi','daxing','pingchang','miyun']
stationsStart3 = ['pinggu']
stationsStart4 = ['mentougou']
stationsStart5 = ['tongzhou','huairou','fengtaihuayuan']
stationsStart6 = ['fangshan']
stations = ['dongsi'] # pick one for test
for s in stations :
    #print(s)
    all = pd.read_csv('../final_project_data/mergeBeijing/'+s+'.csv')
    #print(all.shape) ----------------->
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

    # SI ON VEUT CONCAT : SEPARER SELECTION ET NUMPY
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
    '''
    print(X_test[0])
    print(X_test[-1])
    print(X.shape)
    print(y.shape)
    print(X[0]) # tests see right id station
    print(y[0])
    '''
    # OPTION 1 : DECALER EN JOURs
    X_train1, X_val1 = X[startDateTest-300-days*24:startDateTest-100-days*24], X[startDateTest-100-days*24:startDateTest-days*24]
    X_train1_03, X_val1_03 = X_O3[startDateTest-300-days*24:startDateTest-100-days*24], X_O3[startDateTest-100-days*24:startDateTest-days*24]
    X_train1_PM, X_val1_PM = X_PM[startDateTest-300-days*24:startDateTest-100-days*24], X_O3[startDateTest-100-days*24:startDateTest-days*24]
    y_train1, y_val1 = y[startDateTest-300:startDateTest-100], y[startDateTest-100:startDateTest]
    '''
    print(X_train1[0])
    print(X_val1[-1])
    print(y_train1[0])
    print(y_val1[-1])
    print("option 1 : ")
    print(X_train1.shape)
    print(y_train1.shape)
    print(X_val1.shape)
    print(y_val1.shape)
    '''
    # OPTION 2 : DECALER EN HEURE
    X_train2, X_val2 = X[startDateTest-300-1:startDateTest-100-1], X[startDateTest-100-1:startDateTest-1]
    X_train2_O3, X_val2_O3 = X_O3[startDateTest-300-1:startDateTest-100-1], X_O3[startDateTest-100-1:startDateTest-1]
    X_train2_PM, X_val2_PM = X_PM[startDateTest-300-1:startDateTest-100-1], X_PM[startDateTest-100-1:startDateTest-1]
    y_train2, y_val2 = y[startDateTest-300:startDateTest-100], y[startDateTest-100:startDateTest]
    '''
    print(X_train2[0])
    print(X_val2[-1])
    print(y_train2[0])
    print(y_val2[-1])
    print(X_train2.shape)
    print(y_train2.shape)
    print(X_val2.shape)
    print(y_val2.shape)
    '''
    # OPTION 3 : DECALER EN HEURE AVEC SET DE PREDICTION (3) POUR UN Y
    X_train3= np.array([np.array([ X[i],X[i+1],X[i+2]])for i in range(startDateTest-300-3,startDateTest-100-3)]) #0,1,2... 1,2,3.... 197,198,199
    X_train3_O3 = np.array([np.array([ X_O3[i],X_O3[i+1],X_O3[i+2]])for i in range(startDateTest-300-3,startDateTest-100-3)]) #0,1,2... 1,2,3.... 197,198,199
    X_train3_PM = np.array([np.array([ X_PM[i],X_PM[i+1],X_PM[i+2]])for i in range(startDateTest-300-3,startDateTest-100-3)]) #0,1,2... 1,2,3.... 197,198,199
    X_val3 = np.array([np.array([ X[i],X[i+1],X[i+2]])for i in range(startDateTest-100-3,startDateTest-3+1-1)])
    X_val3_O3 = np.array([np.array([ X_O3[i],X_O3[i+1],X_O3[i+2]])for i in range(startDateTest-100-3,startDateTest-3+1-1)])
    X_val3_PM = np.array([np.array([ X_PM[i],X_PM[i+1],X_PM[i+2]])for i in range(startDateTest-100-3,startDateTest-3+1-1)])
    y_train3, y_val3 = y[startDateTest-300:startDateTest-100], y[startDateTest-100:startDateTest]
    '''
    print(X_train3[0])
    print(X_val3[-1])
    print(y_train3[0])
    print(y_val3[-1])
    print(X_train3.shape)
    print(X_val3.shape)
    print(y_train3.shape)
    print(y_val3.shape)
    print(y_train3[-1])
    print(y_val3[0])
    '''
    '''
    X_train4 = np.array([np.append( X[i],np.append(X[i+1],X[i+2]))for i in range(startDateTest-300-3,startDateTest-100-3)])
    X_train4_O3 = np.array([np.append( X_O3[i],np.append(X_O3[i+1],X_O3[i+2]))for i in range(startDateTest-300-3,startDateTest-100-3)]) #0,1,2... 1,2,3.... 197,198,199
    X_val4 = np.array([np.append( X[i],np.append(X[i+1],X[i+2]))for i in range(startDateTest-100-3,startDateTest-3+1-1)])
    X_val4_O3 = np.array([np.append( X_O3[i],np.append(X_O3[i+1],X_O3[i+2]))for i in range(startDateTest-100-3,startDateTest-3+1-1)])
    X_val4_PM = np.array([np.append( X_PM[i],np.append(X_PM[i+1],X_PM[i+2]))for i in range(startDateTest-100-3,startDateTest-3+1-1)])
    X_train4_PM = np.array([np.append( X_PM[i],np.append(X_PM[i+1],X_PM[i+2]))for i in range(startDateTest-300-3,startDateTest-100-3)]) #0,1,2... 1,2,3.... 197,198,199
    '''
    X_train4 = np.array([X[i] for i in range(startDateTest-300-1,startDateTest-100-1)]) #0,1,2... 1,2,3.... 197,198,199
    X_val4 = np.array([X[i] for i in range(startDateTest-100-1,startDateTest-1+1-1)])
    X_train4_PM = np.array([X_PM[i] for i in range(startDateTest-300-1,startDateTest-100-1)])
    X_val4_PM = np.array([X_PM[i] for i in range(startDateTest-100-1,startDateTest-1+1-1)])
    X_train4_O3 = np.array([X_O3[i] for i in range(startDateTest-300-1,startDateTest-100-1)])
    X_val4_O3 = np.array([X_O3[i] for i in range(startDateTest-100-1,startDateTest-1+1-1)])

    y_train4, y_val4 = y[startDateTest-300:startDateTest-100,0], y[startDateTest-100:startDateTest,0]
    y_train5, y_val5 = y[startDateTest-300:startDateTest-100,1], y[startDateTest-100:startDateTest,1]
    y_train6, y_val6 = y[startDateTest-300:startDateTest-100,2], y[startDateTest-100:startDateTest,2]
    '''
    print(X_train4[0])
    print(X_val4[-1])
    print(y_train4[0])
    print(y_val4[-1])
    print(X_train4.shape)
    print(X_val4.shape)
    print(y_train4.shape)
    print(y_val4.shape)
    print(X_train4[-1])
    print(X_val4[0])
    '''

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
    '''
    x = range(len(np.append(y_train6,y_val6)))
    y = np.append(y_train6,y_val6)
    plt.plot(x,y)
    plt.xlabel('utc_time')
    plt.ylabel('O3 Level')
    plt.title('O3 from data training : '+ s)
    plt.show()
    '''

    '''
    x = range(len(y_test_PM25))
    y = y_test_PM25
    plt.plot(x,y)
    plt.xlabel('utc_time')
    plt.ylabel('PM2.5 Level')
    plt.title('PM10 test: '+ s)
    plt.show()
    '''
    '''
    x = range(len(y_O3))
    y = y_O3
    plt.plot(x,y)
    plt.xlabel('utc_time')
    plt.ylabel('O3 Level')
    plt.title('O3 all: '+ s)
    plt.show()
    '''

print("final matrix : ")
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
'''
print(X_train_O3.shape)
print(X_train_PM.shape)
print(X_val_O3.shape)
print(X_val_PM.shape)
print(X_train_app_O3.shape)
print(X_train_app_PM.shape)
print(X_val_app_O3.shape)
print(X_val_app_PM.shape)
print(y_train_app1.shape)
print(y_train_app2.shape)
print(y_train_app3.shape)
print(y_val_app1.shape)
print(y_val_app2.shape)
print(y_val_app3.shape)
'''
print("----------- START TESTS : Linerar Regression ------------")
'''
X_train_l = X_train_app_PM
y_train_l = y_train_app1
#X_test_l = X_test
X_val_l = X_val_app_PM
y_val_l = y_val_app1


reg = LinearRegression().fit(X_train_l,y_train_l)
err = mean_squared_error(y_val_l,reg.predict(X_val_l))

print("mean squared error : ")
print(err)
pred = reg.predict(X_val_l)
for i in range(10) :
    print("test : ", y_val_l[i])
    print("pred = ", pred[i])
'''
print("----------- START TESTS : Gradient Tree Boosting ------------")
# https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
# https://stackabuse.com/gradient-boosting-classifiers-in-python-with-scikit-learn/
'''
ON SEPARERA ON FONCTION PLUS TARD
'''
#TO FIT : n_estimators, learning rate, AND max_depth
'''
print(X_train.shape)
X_train_sparse = sparse.csr_matrix(X_train)
y_train_sparse = sparse.csr_matrix(y_train)
X_val_sparse = sparse.csr_matrix(X_val)
y_val_sparse = sparse.csr_matrix(y_val)
print(X_train_sparse.shape)
print(y_train_sparse.shape)
print(X_val_sparse.shape)
print(y_val_sparse.shape)
'''
min_error = 100000000000000000
'''
n_est = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50,100,150,200,250,300,350,400,450,500,550,600,750,1000,1250,1500,1750])
learning_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06 ,0.07, 0.08, 0.09, 0.1,0.2,0.3,0.4,0.5])
max_depths = np.array([1,2,3,4,5,6,7,8,9,10])
'''
'''
n_est = np.array([50,100,150,200,250,300,350,400])
learning_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06 ,0.07, 0.08, 0.09, 0.1])
max_depths = np.array([1,2,3])

for esti in n_est :
    for lr in learning_rates :
        for depth in max_depths :
            print('esti: ', esti, ' learning_rate : ', lr, ' depth : ', depth)
            est = GradientBoostingRegressor(n_estimators=esti, learning_rate=lr, max_depth=depth, random_state=0, loss='ls').fit(X_train_app_O3, y_train_app3)
            # make cross validation error
            error = mean_squared_error(y_val_app3, est.predict(X_val_app_O3))
            #print(error)
            if error < min_error :
                min_error = error
                best_n_est = esti
                best_lr = lr
                best_depth = depth


print("best mean squared error : ")
print(min_error)
print("with n_estimators : ")
print(best_n_est)
print("with learning_rate : ")
print(best_lr)
print("with max_depth : ")
print(best_depth)
est = GradientBoostingRegressor(n_estimators=best_n_est, learning_rate=best_lr, max_depth=best_depth, random_state=0, loss='ls').fit(X_train_app_O3, y_train_app3)
pred = est.predict(X_val_app_O3) # CHANGE IN TEST

# COMPARER DEUX PLOTS :
# Y_TEST REELS A PREDIRE
# Y_TEST QUE NOUS ON A PREDIT

# Calculate probabilities
#est_prob = est.predict_proba(X_test)
# Calculate confusion matrix
#confusion_est = confusion_matrix(y_test,pred)
#print(confusion_est)


for i in range(10) :
    print("test : ", y_val_app3[i])
    print("pred = ", pred[i])
'''

print("----------- END TESTS Gradient Tree Boosting ------------")

print("----------- START TESTS : SVM / cross validation ------------")
# PAS FIFOU HEIN
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://scikit-learn.org/stable/modules/svm.html
#clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#clf.score(X_test, y_test)
'''
lab_enc = preprocessing.LabelEncoder()
y_train_encoded = lab_enc.fit_transform(column4)

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, data_predict_x, y_train_encoded, cv=2)
print("scores : ")
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''
print("----------- END TESTS SVM / Cross validation ------------")

print("----------- START TESTS : Random Forest ------------")
# https://machinelearningmastery.com/random-forest-ensemble-in-python/
'''
we will evaluate the model using repeated k-fold cross-validation,
with three repeats and 10 folds. We will report the mean absolute error (MAE)
of the model across all repeats and folds. The scikit-learn library makes the
MAE negative so that it is maximized instead of minimized.
This means that larger negative MAE are better and a perfect model has a MAE of 0.
'''
'''
max_score = 0
X, y = data_predict_x, column4
# TO SET PARAMETERS
max_sample = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, None])
max_feature = np.array([1,2]) # defaults to the square root of the number of input features -> augmenter quand tous
n_estimator = np.array([10,50,100, 200, 300, 400, 500]) # sdet tp 100 by default
max_depth = np.array([None,1,2,3,4,5,6,7])

for max_samp in max_sample :
    for max_feat in max_feature :
        for max_dep in max_depth :
            for n in n_estimator :
                model = RandomForestRegressor(max_samples = max_samp, max_features = max_feat, n_estimators = n, max_depth = max_dep)
                cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
                n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
                if mean(n_scores) < max_score :
                    max_score = mean(n_scores)
                    best_n_est = n
                    best_max_sample = max_samp
                    best_max_feature = max_feat
                    best_max_depth = max_dep

print("best n_estimator : ")
print(best_n_est)
print("best max_feature : ")
print(best_max_feature)
print("best max_sample : ")
print(best_max_sample)
print("best max_depth : ")
print(best_max_depth)

print("results cross-validation error w/ best parameters : ")
X, y = data_predict_x, column4
model = RandomForestRegressor(max_samples = best_max_sample, max_features = best_max_feature, n_estimators = best_n_est, max_depth = best_max_depth)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

print("prediciton...")
model = RandomForestRegressor(max_samples = best_max_sample, max_features = best_max_feature, n_estimators = best_n_est, max_depth = best_max_depth)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
mean_sq_err = mean_squared_error(y_test_pred, y_test)
print("Mean squared error : ")
print(mean_sq_err)
print('Prediction[0]: %d' % yhat[0])
print('Should predict : %d' %y_test[0])
print('Prediction[1]: %d' % yhat[1])
print('Should predict : %d' %y_test[1])
'''

print("----------- END TESTS Random Forest ------------")

print("----------- START TEST : GAM (generalized additive model) ------------")
# https://medium.com/just-another-data-scientist/building-interpretable-models-with-generalized-additive-models-in-python-c4404eaf5515
# The common models are LinearGAM, LogisticGAM, PoissonGAM, GammaGAM, InvGuss.

# OMG DES SAUVEURS :
'''
Find the best model requires the tuning of several key parameters including
n_splines, lam, and constraints.
Among them, lam is of great importance to the performance of GAMs.
It controls the strength of the regularization penalty on each term.

yGAM built a grid search function that build a grid to search over multiple lam
values so that the model with the lowest generalized cross-validation (GCV) score.

n_splines refers to the number of splines to use in each of the smooth function that is going to be fitted.

lam is the penalization term that is multiplied to the second derivative in the overall objective function.

constraints is a list of constraints that allows the user to specify whether a
function should have a monotonically constraint. This needs to be a string
in [‘convex’, ‘concave’, ‘monotonic_inc’, ‘monotonic_dec’,’circular’, ‘none’]
'''
'''
# set model : OVER CONSTRAINT, LAM, SPLINES
gam = LinearGAM(n_splines=10).gridsearch(X_train_app, y_train_app1)
#gam = LogisticGAM(constraints=constraints, lam=lam, n_splines=n_splines).fit(X, y)
#gam.summary()

#prediction :
predictions = gam.predict(X_val_app)
print("Mean squared error: {} over {} samples".format(mean_squared_error(y_val_app1, predictions), y_val_app1.shape[0]))

print("y test : ")
print(y_val_app1)
print("predictions : ")

print(predictions)
'''

#y_test=y_test.astype('int')
#predictions=predictions.astype('int')
#print("Log Loss: {} ".format(log_loss(y_test, predictions)))

'''
XX = gam.generate_X_grid(term=1)
plt.rcParams['figure.figsize'] = (28, 8)
fig, axs = plt.subplots(1, len(boston.feature_names[0:6]))
titles = boston.feature_names
for i, ax in enumerate(axs):
    #pdep, confi = gam.partial_dependence(XX, feature=i+1, width=.95)
    pdep, confi = gam.partial_dependence(XX, width=.95)
    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi[0][:, 0], c='grey', ls='--')
    ax.plot(XX[:, i], confi[0][:, 1], c='grey', ls='--')
    ax.set_title(titles[i],fontsize=26)
plt.show()
'''

print("----------- END TESTS GAM ------------")

print("----------- START TESTS Multiclass Neural Network  ------------")
'''
C_mat = data_predict_x.corr()
fig = plt.figure(figsize = (15,15))
'''
'''
for i in range(10) :
    print("test : ", y_test[i])
    print("pred = ", pred[i])
'''
'''
error = 10000

max_iter = [600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200 ]
activation = ['identity', 'logistic', 'tanh', 'relu']

for it in max_iter:
    for acti in activation:
        #regr = MLPRegressor(solver = 'sgd', max_iter = it).fit(X_train_app, y_train_app1)
        regr = MLPRegressor(max_iter = it, activation = acti).fit(X_train_app, y_train_app2)
        err = mean_squared_error(y_val_app2, regr.predict(X_val_app))
        if (err < error):
            error = err
            best_it = it
            best_acti = acti


print('Best min error : ', error)
print('best it : ', best_it)
print('best acti : ', best_acti)
'''
print("----------- END TESTS Multiclass Neural Network  ------------")

print("----------- START TESTS xgboost  ------------")
# https://xgboost.readthedocs.io/en/latest/parameter.html
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

# OPTION 3
'''
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train_app_PM,y_train_app2)

preds = xg_reg.predict(X_val_app_PM)

rmse = np.sqrt(mean_squared_error(y_val_app2, preds))

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
'''

all = pd.read_csv('../final_project_data/mergeBeijing/dongsi.csv')

startDateTest = 10042
endDateTest = 10089

X_test = all.loc[startDateTest:endDateTest-1,['utc_time','temperature','pressure','humidity','wind_direction','wind_speed','PM2.5','PM10','NO2','CO','O3','SO2']].to_numpy()
X_test_O3 = all.loc[startDateTest:endDateTest-1,['temperature','pressure','humidity','wind_direction','wind_speed','PM10','NO2','CO','O3']].to_numpy()
X_test_PMs = all.loc[startDateTest:endDateTest-1,['humidity','wind_direction','wind_speed','PM2.5','PM10','NO2','CO','O3','SO2']].to_numpy()
y_test = all.loc[startDateTest+1:endDateTest,['utc_time','PM2.5','PM10','O3']].to_numpy()
y_test_PM25 = y_test[:,0]
y_test_PM10 = y_test[:,1]
y_test_O3 = y_test[:,2]

X = all.loc[:startDateTest-1,['utc_time','temperature','pressure','humidity','wind_direction','wind_speed','PM2.5','PM10','NO2','CO','O3','SO2']].to_numpy()
X_O3 = all.loc[:startDateTest-1,['temperature','pressure','humidity','wind_direction','wind_speed','PM10','NO2','CO','O3']].to_numpy()
X_PM = all.loc[:startDateTest-1,['humidity','wind_direction','wind_speed','PM2.5','PM10','NO2','CO','O3','SO2']].to_numpy()
y = all.loc[1:startDateTest,['utc_time','PM2.5','PM10','O3']].to_numpy()
y_PM5 = y[:,0]
y_PM10 = y[:,1]
y_O3 = y[:,2]

print(X_test[0])
print(y_test[0])
print(X[-1])
print(y[-1])


data_dmatrix = xgb.DMatrix(data=X_PM,label=y_PM10/1000)
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
                    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
                    #print(cv_results.head())
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


#xg_reg = xgb.XGBRegressor(objective= 'binary:logistic', colsample_bytree= 0.9, learning_rate= 0.2, max_depth=2, subsample= 0.9,alpha= 10) # FOR PM10
xg_reg = xgb.XGBRegressor(objective= best_objective, colsample_bytree= best_colysample_bytree, learning_rate= best_lr, max_depth=best_depth, subsample= best_sub,alpha= 10) # FOR y_PM10
xg_reg.fit(X_PM,y_PM10/1000)
pred = xg_reg.predict(X_PM)
pred = pred*1000
rmse = np.sqrt(mean_squared_error(y_PM10, pred))
print("mse : ", rmse)
for i in range(10) :
    print("test : ", y_PM[i])
    print("pred = ", pred[i])
# OPTION 5


print("----------- END TESTS xgboost  ------------")

'''
TODO :
- cross validation everywhere
'''
