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
'''
x = aqi_forecast['utc_time']
y = aqi_forecast['PM2.5 (ug-m3)']
x = x[141602:141612]
y = y[141602:141612]
plt.plot(x,y)
plt.xlabel('utc_time')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5  from 2018/3/28 14:00 to 2018/3/29 0:00 , before interpolation')
plt.show()
'''
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
'''
x = aqi_forecast['utc_time']
y = aqi_forecast['PM2.5 (ug-m3)']
x = x[141602:141612]
y = y[141602:141612]
plt.plot(x,y)
plt.xlabel('utc_time')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5  from 2018/3/28 14:00 to 2018/3/29 0:00 , after interpolation')
plt.show()

# PRINT TO SEE OVER TWO WEEKS (to see if day influence)
x = aqi_forecast['utc_time']
y = aqi_forecast['PM2.5 (ug-m3)']
x = x[113031:141612]
y = y[113031:141612]
plt.plot(x,y)
plt.xlabel('utc_time')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 from 2017/06/05 00:00:00 to 2017/06/18 23:00:00 (over two weeks)')
plt.show()

x = aqi_forecast['utc_time']
y = aqi_forecast['PM10 (ug-m3)']
x = x[113031:141612]
y = y[113031:141612]
plt.plot(x,y)
plt.xlabel('utc_time')
plt.ylabel('PM10 Level')
plt.title('PM10 from 2017/06/05 00:00:00 to 2017/06/18 23:00:00 (over two weeks)')
plt.show()

x = aqi_forecast['utc_time']
y = aqi_forecast['NO2 (ug-m3)']
x = x[113031:141612]
y = y[113031:141612]
plt.plot(x,y)
plt.xlabel('utc_time')
plt.ylabel('NO2 Level')
plt.title('NO2 from 2017/06/05 00:00:00 to 2017/06/18 23:00:00 (over two weeks)')
plt.show()

# PRINT TO SEE OVER ONE YEAR (too see if th month influences)
x = aqi_forecast['utc_time']
y = aqi_forecast['PM2.5 (ug-m3)']
x = x[:139528]
y = y[:139528]
plt.plot(x,y)
plt.xlabel('utc_time')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 over one year : 2017')
plt.show()

x = aqi_forecast['utc_time']
y = aqi_forecast['PM10 (ug-m3)']
x = x[:139528]
y = y[:139528]
plt.plot(x,y)
plt.xlabel('utc_time')
plt.ylabel('PM10 Level')
plt.title('PM10 over one year : 2017')
plt.show()

x = aqi_forecast['utc_time']
y = aqi_forecast['NO2 (ug-m3)']
x = x[:139528]
y = y[:139528]
plt.plot(x,y)
plt.xlabel('utc_time')
plt.ylabel('NO2 Level')
plt.title('NO2 over one year : 2017')
plt.show()
'''
print("")
