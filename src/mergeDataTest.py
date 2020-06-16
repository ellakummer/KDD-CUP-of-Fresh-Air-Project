import csv
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# https://medium.com/mongolian-data-stories/ulaanbaatar-air-pollution-part-1-35e17c83f70b (ici plus discret mdr)
# https://medium.com/mongolian-data-stories/part-3-the-model-b2fb9a25a07c
# https://github.com/robertritz/Ulaanbaatar-PM2.5-Prediction/blob/master/Classification%20Model/Ulaanbaatar%20PM2.5%20Prediction%20-%20Classification.ipynb

# ALSO :
# see how to VISUALIZE the datas
# e.g. plot PMs per month ( overall visualisation, avoir une idée du truc - > dans intro rapport ? )
# e.g. afficher PMs en fonction de chaque autre donnée qu'on a (un plot par donnée externe)

# ------- LOAD array London_historical_aqi_forecast_stations_20180331 ------

print(" -------------------- LOAD London_historical_aqi_forecast_stations_20180331")
aqi_forecast = pd.read_csv('../final_project_data/London_historical_aqi_forecast_stations_20180331.csv')

aqi_forecast = aqi_forecast.interpolate(method ='polynomial', order = 2, limit_direction ='forward')
aqi_forecast = aqi_forecast.fillna(aqi_forecast.mean())

print("test shape aqi_forecast: ")
print(aqi_forecast.shape)

C_mat_aqi_forecast = aqi_forecast.corr()
print(C_mat_aqi_forecast)
print(aqi_forecast)

#aqi_forecast = aqi_forecast.to_numpy()
#print(aqi_forecast)
#print(aqi_forecast.shape)
print("")
# ------- LOAD London_historical_aqi_other_stations_20180331 ------
print(" -------------------- LOAD London_historical_aqi_other_stations_20180331")
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
#------- LOAD London_AirQuality_Stations ------
print("-------------------- LOAD London_AirQuality_Stations")
airQuality_stations = pd.read_csv('../final_project_data/London_AirQuality_Stations.csv')

print("test shape airQuality_stations: ")
print(airQuality_stations.shape)

print(airQuality_stations)
'''
print("")
airQuality_stations = airQuality_stations.to_numpy()
print("7th  elements : ")
print(airQuality_stations[7])
'''
print("")
# ------- LOAD London_grid_weather_station ------
print(" -------------------- LOAD London_grid_weather_station")
grid_stations = pd.read_csv('../final_project_data/London_grid_weather_station.csv')

print("test shape grid_stations: ")
print(grid_stations.shape)

print(grid_stations)

print("")
# ------- LOAD London_historical_meo_grid ------
print(" -------------------- LOAD London_historical_meo_grid")
meo_grid = pd.read_csv('../../London_historical_meo_grid.csv')
# TAKES A LOT OF TIME :
#meo_grid = meo_grid.interpolate(method ='polynomial', order = 2, limit_direction ='forward')
#meo_grid = meo_grid.fillna(meo_grid.mean())

print("test shape meo_grid: ")
print(meo_grid.shape)

C_mat_meo_grid = meo_grid.corr()
print(C_mat_meo_grid)
print(meo_grid)

print("-------------------------- MERGE DATAS  --------------------------")
# so we have : aqi_forecast, aqi_other, airQuality_stations, grid_stations, meo_grid
# TRY TAKE FOR ONE STATION : BL0
# BASIC INFOS (IDs)

# -- ICI ON ITERER SUR LES IDs :
# bl0_airQuality_stations ->  id YYY :
bl0_airQuality_station = airQuality_stations.loc[airQuality_stations['id']=='BL0',:]
print("bl0_airQuality_station id: ")
print(bl0_airQuality_station)

# ON CHERCHERA DANS FORECAST ET OTHER PUIS CONCAT EN HAUTEUR
#bl0_aqi_forecast <- id YYY
bl0_aqi_forecast = aqi_forecast.loc[aqi_forecast['station_id']=='BL0',:]
print("bl0_aqi_forecast : ")
print(bl0_aqi_forecast)
#print(bl0_aqi_forecast.iloc[[3]])
bl0_aqi_other = aqi_other.loc[aqi_other['station_id']=='BL0',:]
print("bl0_aqi_other : ")
print(bl0_aqi_other)

#bl0_grid_stations : id YYY --> id london_grid_XX
longitude_bl0 = float(f"{bl0_airQuality_station['Longitude'].values[0]:.1f}")
latitude_bl0 = float(f"{bl0_airQuality_station['Latitude'].values[0]:.1f}")
print("longitude bl0 : ")
print(longitude_bl0)
print("latitude bl0 : ")
print(latitude_bl0)
#print(type(longitude_bl0))
#print(type(latitude_bl0))

bl0_grid_station = grid_stations.loc[grid_stations['longitude']==longitude_bl0,:].loc[grid_stations['latitude']==latitude_bl0,:]['stationName'].values[0]
print("bl0_grid_station id : ")
print(bl0_grid_station)


#bl0_meo_grid <- id london_grid_XX
bl0_meo_grid = meo_grid.loc[meo_grid['stationName']==bl0_grid_station,:]
print("bl0_meo_grid: ")
print(bl0_meo_grid)

print("MERGE")
'''
print(bl0_aqi_forecast.columns)
print(bl0_aqi_other.columns
print(bl0_meo_grid.columns)
'''
if bl0_aqi_forecast.empty :
    bl0_merge =pd.merge(left=bl0_aqi_other, right=bl0_meo_grid, left_on='utc_time', right_on='utc_time')
else :
    bl0_merge =pd.merge(left=bl0_aqi_forecast, right=bl0_meo_grid, left_on='utc_time', right_on='utc_time')


print(bl0_merge)
print(bl0_merge.columns)

# SAVE csv :
bl0_merge.to_csv(r'../final_project_data/merge/BL0.csv', index = False)
