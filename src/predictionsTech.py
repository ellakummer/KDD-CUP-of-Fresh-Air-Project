import csv
import numpy as np
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

column1 = np.empty((0,1))
column2 = np.empty((0,1))
column3 = np.empty((0,1))
column4 = np.empty((0,1))
column5 = np.empty((0,1))
data_predict_x = np.empty((0,2))

with open('../final_project_data/London_historical_aqi_forecast_stations_20180331.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        #print(', '.join(row))
        if line_count != 0 :
            column1 = np.append(column1, np.array([[row[1]]]))
            column2 = np.append(column2, np.array([[row[2]]]))
            column3 = np.append(column3, np.array([[float(row[3])]]))
            column4 = np.append(column4, np.array([[float(row[4])]]))
            column5 = np.append(column5, np.array([[float(row[5])]]))

            data_predict_x = np.append(data_predict_x, np.array([[float(row[3]), float(row[5])]]), axis = 0)

        line_count += 1
        #if line_count == 100000 : #test almost all
        #if line_count == 10800 : #test first id
        #if line_count == 11 : #test small
        if line_count == 301 : #test ok for first id
            break


print("----------- TESTS DATAS ------------")

print("COMBINE : ")
# https://numpy.org/doc/stable/user/basics.rec.html
records = np.rec.fromarrays((column1, column2, column3, column4, column5), names=('date', 'id', 'PM2.5', 'PM10', 'N02'))
'''
print(records)
print("DATE : ")
print(records['date'])
print("N02 : ")
print(records['N02'])
print("PM2.5 : ")
print(records['PM2.5'])
'''

print("#datas : ", line_count-1)
print("test shape : ")
print(records.shape)

print("7th element : ")
print(records[7])
print("9th element : ")
print(records[9])
print("1st parameter 9th element : ")
print(records[9][0])
print("3rd parameter 9th element : ")
print(records[9][2])
print("last element : ")
print(records[-1])
print("test sum :")
print(records[7][2] + records[9][2])

print("----------- END TEST DATAS ------------")

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
add second datas : London_historical_aqi_other_stations_20180331
'''

X_train, X_test = data_predict_x[:200], data_predict_x[200:] # we use PM2.5 and N02
y_train, y_test = column4[:200], column4[200:] # we predict PM10

print("test to use shape: ")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


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
            error = mean_squared_error(y_test, est.predict(X_test))
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
# Calculate probabilities
#est_prob = est.predict_proba(X_test)
# Calculate confusion matrix
#confusion_est = confusion_matrix(y_test,pred)
#print(confusion_est)
'''
'''
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
