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
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
#from xgboost import XGBRegressor
'''
# tests
from sklearn.datasets import load_boston
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

X_train = np.empty([198,3,8])
y_train = np.empty([198,2])
X_val = np.empty([100,3,8])
y_val = np.empty([100,2])

days = 2
stations = ['BL0', 'BX1', 'BX9', 'CD1', 'CD9', 'CR8', 'CT2', 'CT3', 'GB0', 'GN0', 'GN3', 'GR4', 'GR9', 'HR1', 'HV1', 'KC1', 'KF1', 'LH0', 'LW2', 'MY7', 'RB7', 'ST5', 'TD5', 'TH4']
#stations = ['BL0'] # pick one
for s in stations :
    print(s)
    all = pd.read_csv('../final_project_data/merge/'+s+'.csv')
    print(all.shape)
    if s == 'CT2' :
        startDate = 9240
        endDate = 9287
    else :
        startDate = 10656
        endDate = 10703
    # SEE YOU LATER TESTS :)
    #X_test = all.loc[startDate:endDate,['temperature','pressure','humidity','wind_direction','wind_speed/kph','PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
    #y_test = all.loc[startDate:endDate,['PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
    X = all[['temperature','pressure','humidity','wind_direction','wind_speed/kph','PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
    y = all[['PM2.5 (ug-m3)','PM10 (ug-m3)']].to_numpy()
    # OPTION 1 : DECALER EN JOURs
    X_train1, X_val1 = X[:200], X[200:300]
    y_train1, y_val1 = y[days*24:200+days*24], y[200+days*24:300+days*24]
    '''
    print("option 1 : ")
    print(X_train1.shape)
    print(y_train1.shape)
    print(X_val1.shape)
    print(y_val1.shape)
    '''
    # OPTION 2 : DECALER EN HEURE
    print("option 2 : ")
    X_train2, X_val2 = X[:200], X[200:300]
    y_train2, y_val2 = y[1:201], y[201:301]
    '''
    print(X_train2.shape)
    print(y_train2.shape)
    print(X_val2.shape)
    print(y_val2.shape)
    '''
    # OPTION 3 : DECALER EN HEURE AVEC SET DE PREDICTION (3) POUR UN Y
    print("option 3 : ")
    X_train3 = np.array([np.array([ X[i],X[i+1],X[i+2]])for i in range(200-3+1)]) #0,1,2... 1,2,3.... 197,198,199
    X_val3 = np.array([np.array([ X[i],X[i+1],X[i+2]])for i in range(198,300-3+1)])
    y_train3, y_val3 = y[3:201], y[201:301]
    '''
    print(X_train3[0])
    print(X_val3[0])
    print(X_train3.shape)
    print(y_train3.shape)
    print(X_val3.shape)
    print(y_val3.shape)
    '''
    print("concat : ")
    X_train = np.append(X_train, X_train3,axis=0)
    y_train = np.append(y_train, y_train3,axis=0)
    X_val = np.append(X_val, X_val3,axis=0)
    y_val = np.append(y_val, y_val3,axis=0)

print("final matrix : ")
X_train = X_train[198:]
y_train = y_train[198:]
X_val= X_val[100:]
y_val = y_val[100:]

print("----------- START TESTS : Linerar Regression ------------")


print("----------- START TESTS : Gradient Tree Boosting ------------")
# https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting
'''
ON SEPARERA ON FONCTION PLUS TARD
'''
#TO FIT : n_estimators, learning rate, AND max_depth
'''
min_error = 1000

n_est = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50,100,150,200,250,300,350,400,450,500,550,600,750,1000,1250,1500,1750])
learning_rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06 ,0.07, 0.08, 0.09, 0.1,0.2,0.3,0.4,0.5])
max_depths = np.array([1,2,3,4,5,6,7,8,9,10])

n_est = np.array([1, 2, 3])
learning_rates = np.array([0.01, 0.02, 0.03])
max_depths = np.array([1,2])

for esti in n_est :
    for lr in learning_rates :
        for depth in max_depths :
            est = GradientBoostingRegressor(n_estimators=esti, learning_rate=lr, max_depth=depth, random_state=0, loss='ls').fit(X_train, y_train)
            # make cross validation error
            error = mean_squared_error(y_val, est.predict(X_val))
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
est = GradientBoostingRegressor(n_estimators=best_n_est, learning_rate=best_lr, max_depth=best_depth, random_state=0, loss='ls').fit(X_train, y_train)
pred = est.predict(X_test)

# COMPARER DEUX PLOTS :
# Y_TEST REELS A PREDIRE
# Y_TEST QUE NOUS ON A PREDIT

# Calculate probabilities
#est_prob = est.predict_proba(X_test)
# Calculate confusion matrix
#confusion_est = confusion_matrix(y_test,pred)
#print(confusion_est)


for i in range(10) :
    print("test : ", y_test[i])
    print("pred = ", pred[i])
'''

# OR :
'''
# https://www.datacareer.ch/blog/parameter-tuning-in-gradient-boosting-gbm-with-python/
p_test2 = {'max_depth':[2,3,4,5,6,7] }
p_test3 = {'max_depth':[2,3,4,5,6,7], 'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750]}

tuning = GridSearchCV(estimator =GradientBoostingRegressor(random_state=0, loss = 'ls'),
            param_grid = p_test3, scoring='neg_mean_squared_error', cv=5)
tuning.fit(X_train,y_train)
print(tuning.best_params_, tuning.best_score_)
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
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
target_df = pd.Series(boston.target)
df.head()

X = df.fillna(df.mean()) # combined X = df, X = X.fillna(X.mean())
y = target_df.fillna(target_df.mean())
X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# set model : OVER CONSTRAINT, LAM, SPLINES
gam = LinearGAM(n_splines=10).gridsearch(X_train, y_train)
#gam = LogisticGAM(constraints=constraints, lam=lam, n_splines=n_splines).fit(X, y)
#gam.summary()

#prediction :
predictions = gam.predict(X_test)
print("Mean squared error: {} over {} samples".format(mean_squared_error(y_test, predictions), y.shape[0]))

print("y test : ")
print(y_test)
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

print("----------- END TESTS Multiclass Neural Network  ------------")


'''
TODO :
- cross validation everywhere
'''
