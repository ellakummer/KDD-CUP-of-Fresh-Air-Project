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
print("-------------------- LOAD beijing_17_18_meo")
beijing_17_18_meo = pd.read_csv('../final_project_data/beijing_17_18_meo_nocoord.csv')

print("test shape beijing_17_18_meo: ")
print(beijing_17_18_meo.shape)

C_mat_beijing_17_18_meo = beijing_17_18_meo.corr()
print(C_mat_beijing_17_18_meo)
print(beijing_17_18_meo)
print(beijing_17_18_meo.columns)


print("")
# ------- LOAD beijing_201802_201803_me ------
print(" -------------------- LOAD beijing_201802_201803_me")
beijing_201802_201803_me = pd.read_csv('../final_project_data/beijing_201802_201803_me.csv')

print("test shape beijing_201802_201803_me: ")
print(beijing_201802_201803_me.shape)

C_mat_beijing_201802_201803_me= beijing_201802_201803_me.corr()
print(C_mat_beijing_201802_201803_me)
print(beijing_201802_201803_me)
print(beijing_201802_201803_me.columns)


print("")
print("-------------------------- CONCAT DATAS (1/2 + 2/2) --------------------------")

beijing_aq = pd.concat([beijing_17_18_aq,beijing_201802_201803_aq])
beijing_meo = pd.concat([beijing_17_18_meo,beijing_201802_201803_me])

print(beijing_aq.shape)
print(beijing_aq.columns)
print(beijing_meo.shape)
print(beijing_meo.columns)

# fangshan
'''
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

#print(bl0_aqi_forecast.columns)
#print(bl0_aqi_other.columns
#print(bl0_meo_grid.columns)

if bl0_aqi_forecast.empty :
    bl0_merge =pd.merge(left=bl0_aqi_other, right=bl0_meo_grid, left_on='utc_time', right_on='utc_time')
else :
    bl0_merge =pd.merge(left=bl0_aqi_forecast, right=bl0_meo_grid, left_on='utc_time', right_on='utc_time')


print(bl0_merge)
print(bl0_merge.columns)

# SAVE csv :
bl0_merge.to_csv(r'../final_project_data/mergeBeijing/BL0.csv', index = False)
'''
