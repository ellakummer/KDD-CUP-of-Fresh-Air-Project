import csv
import numpy as np
import sklearn
import pandas as pd

days = 2


airQuality_stations = pd.read_csv('../final_project_data/London_AirQuality_Stations.csv')
# ../final_project_data/merge/'+station+'.csv
for station in airQuality_stations['id']:


BL0_merge = pd.read_csv('../final_project_data/merge/BL0.csv')
print(BL0_merge)

print("-------------------------- CONVERT TO NUMPY --------------------------")
# WE HAVE :
# 'utc_time', 'station_id', 'PM2.5 (ug-m3)', 'PM10 (ug-m3)','NO2 (ug-m3)', 'stationName',
# 'longitude', 'latitude', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed/kph'
# WE WANT :
# X to predict : temperature,pressure,humidity,wind_direction,wind_speed/kph, AND 'PM2.5 (ug-m3)', 'PM10 (ug-m3)','NO2 (ug-m3)'
# y to predict : PM2.5 (ug-m3) | PM10 (ug-m3)|  NO2 (ug-m3)
X = BL0_merge[['temperature','pressure','humidity','wind_direction','wind_speed/kph','PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
y = BL0_merge[['PM2.5 (ug-m3)','PM10 (ug-m3)','NO2 (ug-m3)']].to_numpy()
print("X and y tests : ")
print(X.shape)
print(y.shape)

print("-------------------------- CREATE TESTS DATA --------------------------")
X_train, X_val = X[:200], X[200:300]
y_train, y_val = y[days:200+days], y[200+days:300+days]

print("tests shape : ")
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
print("tests shift")
print("X 0,1,2,3")
print(X_train[0])
print(X_train[1])
print(X_train[2])
print(X_train[3])
print("y 0,1,2,3")
print(y_train[0])
print(y_train[1])
print(y_train[2])
print(y_train[3])
print("must be the same : X 2,3 and y 0,1")
