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
from xgboost import XGBRegressor
'''
# tests
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# plots
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


BL0_merge = pd.read_csv('../final_project_data/merge/BL0.csv')

print(BL0_merge)

print("-------------------------- CONVERT TO NUMPY --------------------------")
# take less values for TESTS
# Read in first 300 lines of table
#bl0_merge_test = bl0_merge.head(300)
X = BL0_merge[['temperature','pressure','humidity','wind_direction','wind_speed/kph','PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']]
y = BL0_merge[['PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']]
print(X)
print(X.columns)
print(y)
print(y.columns)

BL0_merge = BL0_merge.to_numpy()

#X = X.head(300+days)
#y = y.head(300+days)
X = X.to_numpy()
y = y.to_numpy()
print("X and y tests : ")
print(X.shape)
print(y.shape)


print("-------------------------- CREATE TESTS DATA --------------------------")
# WE HAVE :
# 'utc_time', 'station_id', 'PM2.5 (ug-m3)', 'PM10 (ug-m3)','NO2 (ug-m3)', 'stationName',
# 'longitude', 'latitude', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph'
# WE WANT :
# X to predict : temperature,pressure,humidity,wind_direction,wind_speed/kph, AND 'PM2.5 (ug-m3)', 'PM10 (ug-m3)','NO2 (ug-m3)'
# y to predict : PM2.5 (ug-m3) | PM10 (ug-m3)|  NO2 (ug-m3)

#X = bl0_merge[['temperature','pressure','humidity','wind_direction','wind_speed/kph','PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']]
#Y = bl0_merge[['PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']]
X_train, X_test = X[:200], X[200:300] # we use PM2.5 and N02
y_train, y_test = y[:200], y[200:300] # we predict PM10

print("test to use shape: ")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
