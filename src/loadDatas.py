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


# https://medium.com/mongolian-data-stories/ulaanbaatar-air-pollution-part-1-35e17c83f70b (ici plus discret mdr)
# https://medium.com/mongolian-data-stories/part-3-the-model-b2fb9a25a07c
# https://github.com/robertritz/Ulaanbaatar-PM2.5-Prediction/blob/master/Classification%20Model/Ulaanbaatar%20PM2.5%20Prediction%20-%20Classification.ipynb

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


# ------- LOAD array London_historical_aqi_forecast_stations_20180331 ------

print("LOAD London_historical_aqi_forecast_stations_20180331")
aqi_forecast = pd.read_csv('../final_project_data/London_historical_aqi_forecast_stations_20180331.csv')

aqi_forecast = aqi_forecast.interpolate(method ='polynomial', order = 2, limit_direction ='forward')
aqi_forecast = aqi_forecast.fillna(aqi_forecast.mean())

print("test shape aqi_forecast: ")
print(aqi_forecast.shape)

C_mat_aqi_forecast = aqi_forecast.corr()
print(C_mat_aqi_forecast)
print(aqi_forecast)
'''
x = aqi_forecast['MeasurementDateGMT']
y = aqi_forecast['PM2.5 (ug/m3)']
x = x[141602:141612]
y = y[141602:141612]
plt.plot(x,y)
plt.xlabel('MeasurementDateGMT')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5  from 2018/3/28 14:00 to 2018/3/29 0:00 , before interpolation')
plt.show()
'''
print(aqi_forecast.iloc[[3]])


#aqi_forecast = aqi_forecast.to_numpy()
#print(aqi_forecast)
#print(aqi_forecast.shape)
'''
x = aqi_forecast['MeasurementDateGMT']
y = aqi_forecast['PM2.5 (ug/m3)']
x = x[141602:141612]
y = y[141602:141612]
plt.plot(x,y)
plt.xlabel('MeasurementDateGMT')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5  from 2018/3/28 14:00 to 2018/3/29 0:00 , after interpolation')
plt.show()
'''
print("")
# ------- LOAD London_historical_aqi_other_stations_20180331 ------
print("LOAD London_historical_aqi_other_stations_20180331")
aqi_other = pd.read_csv('../final_project_data/London_historical_aqi_other_stations_20180331.csv')
'''
print("aqi_other lines : ")
print(aqi_other.iloc[[3]])
print("test shape aqi_other: ")
print(aqi_other.shape)
print(aqi_forecast.dtypes)
print(aqi_other.dtypes)
'''
aqi_other = aqi_other.interpolate(method ='polynomial', order = 2, limit_direction ='forward')
aqi_other = aqi_other.fillna(aqi_other.mean())

C_mat_aqi_other = aqi_other.corr()
print(C_mat_aqi_other)
print(aqi_other)
'''
print("")
aqi_other = aqi_other.to_numpy()
print("7th and 9th elements : ")
print(aqi_other[7])
print(aqi_other[9])
print("test sum :")
print(aqi_other[7][3] + aqi_other[9][3])
'''

print("")
'''
#------- LOAD London_AirQuality_Stations ------
print("LOAD London_AirQuality_Stations")
airQuality_stations = pd.read_csv('../final_project_data/London_AirQuality_Stations.csv')

print("test shape airQuality_stations: ")
print(airQuality_stations.shape)

C_mat_airQuality_stations= airQuality_stations.corr()
print(C_mat_airQuality_stations)
print(airQuality_stations)

print("")
# ------- LOAD London_grid_weather_station ------
print("LOAD London_grid_weather_station")
grid_stations = pd.read_csv('../final_project_data/London_grid_weather_station.csv')

print("test shape grid_stations: ")
print(grid_stations.shape)

C_mat_grid_stations = grid_stations.corr()
print(C_mat_grid_stations)
print(grid_stations)

print("")
# ------- LOAD London_historical_meo_grid ------
print("LOAD London_historical_meo_grid")
meo_grid = pd.read_csv('../../London_historical_meo_grid.csv')

print("test shape meo_grid: ")
print(aqi_forecast.shape)

C_mat_meo_grid = meo_grid.corr()
print(C_mat_meo_grid)
print(meo_grid)
'''

''' -------------------------- CONVERT TO NUMPY ----------------------------'''

aqi_forecast = aqi_forecast.to_numpy()
# select column 3 and 5 (PM2.5 and O3)
data_predict_x_aqi_forecast = aqi_forecast[:, [3, 5]]
# select column 4 : PM10
data_predict_y_aqi_forecast = aqi_forecast[:, 4]

print("TEST NUMPY DATAS (aqi forecast) : ")
print(data_predict_x_aqi_forecast)
print(data_predict_y_aqi_forecast)
print(data_predict_x_aqi_forecast.shape)
print(data_predict_y_aqi_forecast.shape)
print(data_predict_y_aqi_forecast)

print("---------------------- TESTS DATAS ---------------------------------")

print("COMBINE : ")
# https://numpy.org/doc/stable/user/basics.rec.html
'''
print(records)
print("DATE : ")
print(records['date'])
print("N02 : ")
print(records['N02'])
print("PM2.5 : ")
print(records['PM2.5'])
'''

print(aqi_forecast[141609])  # column filled  by interpolation
print("7th element : ")
print(aqi_forecast[7])
print("9th element : ")
print(aqi_forecast[9])
print("1st parameter 9th element : ")
print(aqi_forecast[9][0])
print("3rd parameter 9th element : ")
print(aqi_forecast[9][2])
print("last element : ")
print(aqi_forecast[-1])
print("test sum :")
print(aqi_forecast[7][3] + aqi_forecast[9][3])

print("----------- END TEST DATAS ------------")

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
add second datas : London_historical_aqi_other_stations_20180331
'''

X_train, X_test = data_predict_x_aqi_forecast[:200], data_predict_x_aqi_forecast[200:300] # we use PM2.5 and N02
y_train, y_test = data_predict_y_aqi_forecast[:200], data_predict_y_aqi_forecast[200:300] # we predict PM10

print("test to use shape: ")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
