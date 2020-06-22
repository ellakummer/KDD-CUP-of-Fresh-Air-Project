import csv
import numpy as np
import sklearn
import pandas as pd

# ALSO :
# see how to VISUALIZE the datas
# e.g. plot PMs per month ( overall visualisation, avoir une idée du truc - > dans intro rapport ? )
# e.g. afficher PMs en fonction de chaque autre donnée qu'on a (un plot par donnée externe)

print("------------------------------ LOAD DATA  ------------------------------")

# ------- LOAD array London_historical_aqi_forecast_stations_20180331 ------

print(" -------------------- LOAD London_historical_aqi_forecast_stations_20180331")

aqi_forecast = pd.read_csv('../final_project_data/London_historical_aqi_forecast_stations_20180331(copie).csv')
#aqi_forecast = aqi_forecast.interpolate(method ='polynomial', order = 2, limit_direction ='forward')
# aqi_forecast = aqi_forecast.interpolate(method ='linear')
# aqi_forecast = aqi_forecast.fillna(aqi_forecast.mean())

print("test shape aqi_forecast: ")
print(aqi_forecast.shape)
C_mat_aqi_forecast = aqi_forecast.corr()
print("correlation matrix : ")
print(C_mat_aqi_forecast)
print("Data aqi_forecast : ")
print(aqi_forecast)

print("")
# ------- LOAD London_historical_aqi_other_stations_20180331 ------
print(" -------------------- LOAD London_historical_aqi_other_stations_20180331")

aqi_other = pd.read_csv('../final_project_data/London_historical_aqi_other_stations_20180331.csv')
#aqi_other = aqi_other.interpolate(method ='polynomial', order = 2, limit_direction ='forward')
# aqi_other = aqi_other.interpolate(method ='linear')
# aqi_other = aqi_other.fillna(aqi_other.mean())

print("correlation matrix : ")
C_mat_aqi_other = aqi_other.corr()
print(C_mat_aqi_other)
print("Data aqi_other : ")
print(aqi_other)

print("")
#------- LOAD London_AirQuality_Stations ------
print("-------------------- LOAD London_AirQuality_Stations")

airQuality_stations = pd.read_csv('../final_project_data/London_AirQuality_Stations.csv')

print("test shape airQuality_stations: ")
print(airQuality_stations.shape)
print("Data airQuality_stations : ")
print(airQuality_stations)

print("")
# ------- LOAD London_grid_weather_station ------
print(" -------------------- LOAD London_grid_weather_station")

grid_stations = pd.read_csv('../final_project_data/London_grid_weather_station.csv')
'''
print("test shape grid_stations: ")
print(grid_stations.shape)
print("Data grid_stations : ")
print(grid_stations)
'''
print("")
# ------- LOAD London_historical_meo_grid ------
print(" -------------------- LOAD London_historical_meo_grid")

meo_grid = pd.read_csv('../../London_historical_meo_grid.csv')
# TAKES A LOT OF TIME :
#meo_grid = meo_grid.interpolate(method ='polynomial', order = 2, limit_direction ='forward')
#meo_grid = meo_grid.fillna(meo_grid.mean())

print("test shape meo_grid: ")
print(meo_grid.shape)
print("Correlation matrix : ")
C_mat_meo_grid = meo_grid.corr()
print(C_mat_meo_grid)
print("Data meo_grid : ")
print(meo_grid)

print("")
print("-------------------------- MERGE DATAS  --------------------------")
for station in airQuality_stations['id']:
    print("Station : ")
    print(station)

    station_airQuality_station = airQuality_stations.loc[airQuality_stations['id']==station,:]

    #station_aqi_forecast <- id YYY
    station_aqi_forecast = aqi_forecast.loc[aqi_forecast['station_id']==station,:]
    print("station_aqi_forecast : ")
    print(station_aqi_forecast)

    station_aqi_other = aqi_other.loc[aqi_other['station_id']==station,:]
    print("station_aqi_other : ")
    print(station_aqi_other)

    #station_grid_stations : id YYY --> id london_grid_XX (w/ longitude + latitude)
    longitude_station = float(f"{station_airQuality_station['Longitude'].values[0]:.1f}")
    latitude_station = float(f"{station_airQuality_station['Latitude'].values[0]:.1f}")
    print("longitude station : ")
    print(longitude_station)
    print("latitude station : ")
    print(latitude_station)

    station_grid_station = grid_stations.loc[grid_stations['longitude']==longitude_station,:].loc[grid_stations['latitude']==latitude_station,:]['stationName'].values[0]
    print("station_grid_station id : ")
    print(station_grid_station)

    #station_meo_grid <- id london_grid_XX
    station_meo_grid = meo_grid.loc[meo_grid['stationName']==station_grid_station,:]
    print("station_meo_grid: ")
    print(station_meo_grid)

    print("MERGE")

    print(station_aqi_forecast.columns)
    print(station_aqi_other.columns)
    print(station_meo_grid.columns)

    if station_aqi_forecast.empty :
        station_merge =pd.merge(left=station_aqi_other, right=station_meo_grid, left_on='utc_time', right_on='utc_time')
    else :
        station_merge =pd.merge(left=station_aqi_forecast, right=station_meo_grid, left_on='utc_time', right_on='utc_time')

    station_merge = station_merge.interpolate(method ='linear')
    station_merge = station_merge.fillna(station_merge.mean())

    print(station_merge)
    print(station_merge.columns)

    # SAVE csv :
    station_merge.to_csv(r'../final_project_data/merge/'+station+'.csv', index = False)
