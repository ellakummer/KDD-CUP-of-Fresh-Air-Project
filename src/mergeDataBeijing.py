import csv
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# ALSO :
# see how to VISUALIZE the datas
# e.g. plot PMs per month ( overall visualisation, avoir une idée du truc - > dans intro rapport ? )
# e.g. afficher PMs en fonction de chaque autre donnée qu'on a (un plot par donnée externe)

# ------- LOAD array beijing_17_18_aq ------

print(" -------------------- LOAD beijing_17_18_aq")
beijing_17_18_aq = pd.read_csv('../final_project_data/beijing_17_18_aq.csv')

beijing_17_18_aq = beijing_17_18_aq.interpolate(method ='polynomial', order = 2, limit_direction ='forward')
beijing_17_18_aq = beijing_17_18_aq.fillna(beijing_17_18_aq.mean())

print("test shape beijing_17_18_aq: ")
print(beijing_17_18_aq.shape)

C_mat_beijing_17_18_aq = beijing_17_18_aq.corr()
print(C_mat_beijing_17_18_aq)
print(beijing_17_18_aq)
print(beijing_17_18_aq.columns)

#aqi_forecast = aqi_forecast.to_numpy()
#print(aqi_forecast)
#print(aqi_forecast.shape)
print("")
# ------- LOAD beijing_201802_201803_aq ------
print(" -------------------- LOAD beijing_201802_201803_aq")
beijing_201802_201803_aq = pd.read_csv('../final_project_data/beijing_201802_201803_aq.csv')

beijing_201802_201803_aq = beijing_201802_201803_aq.interpolate(method ='polynomial', order = 2, limit_direction ='forward')
beijing_201802_201803_aq = beijing_201802_201803_aq.fillna(beijing_201802_201803_aq.mean())

print("test shape beijing_201802_201803_aq: ")
print(beijing_201802_201803_aq.shape)

C_mat_beijing_201802_201803_aq= beijing_201802_201803_aq.corr()
print(C_mat_beijing_201802_201803_aq)
print(beijing_201802_201803_aq)
print(beijing_201802_201803_aq.columns)


print("")
#------- LOAD beijing_17_18_meo ------
print("-------------------- LOAD beijing_17_18_meo_cut")
beijing_17_18_meo = pd.read_csv('../final_project_data/beijing_17_18_meo_cut.csv')

beijing_17_18_meo = beijing_17_18_meo.interpolate(method ='polynomial', order = 2, limit_direction ='forward')
beijing_17_18_meo = beijing_17_18_meo.fillna(beijing_17_18_meo.mean())

print("test shape beijing_17_18_meo: ")
print(beijing_17_18_meo.shape)

C_mat_beijing_17_18_meo = beijing_17_18_meo.corr()
print(C_mat_beijing_17_18_meo)
print(beijing_17_18_meo)
print(beijing_17_18_meo.columns)


print("")
# ------- LOAD beijing_201802_201803_me ------
print(" -------------------- LOAD beijing_201802_201803_me_cut")
beijing_201802_201803_me = pd.read_csv('../final_project_data/beijing_201802_201803_me_cut.csv')

# remove duplicated lines 00:00:00
beijing_201802_201803_me.drop_duplicates(subset =['station_id','utc_time'], keep = 'first', inplace = True)

beijing_201802_201803_me = beijing_201802_201803_me.interpolate(method ='polynomial', order = 2, limit_direction ='forward')
beijing_201802_201803_me = beijing_201802_201803_me.fillna(beijing_201802_201803_me.mean())

print("test shape beijing_201802_201803_me: ")
print(beijing_201802_201803_me.shape)

C_mat_beijing_201802_201803_me= beijing_201802_201803_me.corr()
print(C_mat_beijing_201802_201803_me)
print(beijing_201802_201803_me)
print(beijing_201802_201803_me.columns)

# ------- LOAD Beijing_grid_weather_station ------
print(" -------------------- LOAD Beijing_grid_weather_station")
Beijing_grid_weather_station = pd.read_csv('../final_project_data/Beijing_grid_weather_station.csv')

print("test shape Beijing_grid_weather_station: ")
print(Beijing_grid_weather_station.shape)
print(Beijing_grid_weather_station)
print(Beijing_grid_weather_station.columns)

# ------- LOAD beijing_201802_201803_me ------
print(" -------------------- LOAD Beijing_AirQuality_Stations_en_csv_formated")
Beijing_AirQuality_Stations_en_csv_formated = pd.read_csv('../final_project_data/Beijing_AirQuality_Stations_en_csv_formated.csv')

print("test shape Beijing_AirQuality_Stations_en_csv_formated: ")
print(Beijing_AirQuality_Stations_en_csv_formated.shape)
print(Beijing_AirQuality_Stations_en_csv_formated)
print(Beijing_AirQuality_Stations_en_csv_formated.columns)

# ------- LOAD beijing_201802_201803_me ------

print(" -------------------- LOAD Beijing_historical_meo_grid")
Beijing_historical_meo_grid = pd.read_csv('../../Beijing_historical_meo_grid.csv')

#Beijing_historical_meo_grid = Beijing_historical_meo_grid.interpolate(method ='polynomial', order = 2, limit_direction ='forward')
#Beijing_historical_meo_grid = Beijing_historical_meo_grid.fillna(Beijing_historical_meo_grid.mean())

print("test shape Beijing_historical_meo_grid: ")
print(Beijing_historical_meo_grid.shape)
print(Beijing_historical_meo_grid)
print(Beijing_historical_meo_grid.columns)


print("")
print("-------------------------- CONCAT DATAS (1/2 + 2/2) --------------------------")

beijing_aq = pd.concat([beijing_17_18_aq,beijing_201802_201803_aq])
beijing_meo = pd.concat([beijing_17_18_meo,beijing_201802_201803_me])

print(beijing_aq.shape)
print(beijing_aq.columns)
print(beijing_meo.shape)
print(beijing_meo.columns)
'''
print("-------------------------- SELECT GRID IDs --------------------------")

for s in Beijing_AirQuality_Stations_en_csv_formated['stationId']:
    dongsi_station = Beijing_AirQuality_Stations_en_csv_formated.loc[Beijing_AirQuality_Stations_en_csv_formated['stationId']==s,:]
    print(dongsi_station)
    longitude_station = float(f"{dongsi_station['longitude'].values[0]:.1f}")
    latitude_station = float(f"{dongsi_station['latitude'].values[0]:.1f}")
    print("longitude station : ")
    print(longitude_station)
    print("latitude station : ")
    print(latitude_station)

    station_grid_station = Beijing_grid_weather_station.loc[Beijing_grid_weather_station['longitude']==longitude_station,:].loc[Beijing_grid_weather_station['latitude']==latitude_station,:]['stationName'].values[0]
    print("station_grid_station id : ")
    print(station_grid_station)
'''
'''
print("-------------------------- MERGE DATAS MEO+AQ  --------------------------")

station = 'fangshan'
fangshan_aqi_forecast = beijing_aq.loc[beijing_aq['stationId']==station,:]
fangshan_meo_grid = beijing_meo.loc[beijing_meo['station_id']==station,:]

print(fangshan_aqi_forecast.shape)
print(fangshan_meo_grid.shape)

fangshan_merge =pd.merge(left=fangshan_aqi_forecast, right=fangshan_meo_grid, left_on='utc_time', right_on='utc_time')

print(fangshan_merge)
print(fangshan_merge.columns)

# SAVE csv :
fangshan_merge.to_csv(r'../final_project_data/mergeBeijing/fangshan.csv', index = False)

print("-------------------------- MERGE DATAS AQ + GRID --------------------------")

station = 'dongsi'
dongsi_aqi_forecast = beijing_aq.loc[beijing_aq['stationId']==station,:]
print(dongsi_aqi_forecast.shape)

dongsi_station = Beijing_AirQuality_Stations_en_csv_formated.loc[Beijing_AirQuality_Stations_en_csv_formated['stationId']==station,:]
print(dongsi_station)
longitude_station = float(f"{dongsi_station['longitude'].values[0]:.1f}")
latitude_station = float(f"{dongsi_station['latitude'].values[0]:.1f}")
print("longitude station : ")
print(longitude_station)
print("latitude station : ")
print(latitude_station)

station_grid_station = Beijing_grid_weather_station.loc[Beijing_grid_weather_station['longitude']==longitude_station,:].loc[Beijing_grid_weather_station['latitude']==latitude_station,:]['stationName'].values[0]
print("station_grid_station id : ")
print(station_grid_station)

station_meo_grid = Beijing_historical_meo_grid.loc[Beijing_historical_meo_grid['stationName']==station_grid_station,:]
station_meo_grid = station_meo_grid.fillna(station_meo_grid.mean())
print("station_meo_grid: ")
print(station_meo_grid)

dongsi_merge =pd.merge(left=dongsi_aqi_forecast, right=station_meo_grid, left_on='utc_time', right_on='utc_time')

print(dongsi_merge)
print(dongsi_merge.columns)

# SAVE csv :
fangshan_merge.to_csv(r'../final_project_data/mergeBeijing/dongsi.csv', index = False)
'''

print("-------------------------- MERGE DATAS  --------------------------")
for station in Beijing_AirQuality_Stations_en_csv_formated['stationId']:

    station_aqi_forecast = beijing_aq.loc[beijing_aq['stationId']==station,:]

    station_meo_grid = beijing_meo.loc[beijing_meo['station_id']==station,:]
    if station_meo_grid.empty :
        station_line = Beijing_AirQuality_Stations_en_csv_formated.loc[Beijing_AirQuality_Stations_en_csv_formated['stationId']==station,:]
        longitude_station = float(f"{station_line['longitude'].values[0]:.1f}")
        latitude_station = float(f"{station_line['latitude'].values[0]:.1f}")
        station_grid_name = Beijing_grid_weather_station.loc[Beijing_grid_weather_station['longitude']==longitude_station,:].loc[Beijing_grid_weather_station['latitude']==latitude_station,:]['stationName'].values[0]
        station_meo_grid = Beijing_historical_meo_grid.loc[Beijing_historical_meo_grid['stationName']==station_grid_name,['stationName','utc_time','temperature','pressure','humidity','wind_direction','wind_speed']]
        station_meo_grid = station_meo_grid.fillna(station_meo_grid.mean())

    merge_file =pd.merge(left=station_aqi_forecast, right=station_meo_grid, left_on='utc_time', right_on='utc_time')
    merge_file.to_csv(r'../final_project_data/mergeBeijing/'+station+'.csv', index = False)
