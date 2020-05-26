import csv
import numpy as np

# transform into array

london_aqui_all_stations = np.array([1,1,1,1,1,1])

with open('../final_project_data/London_historical_aqi_forecast_stations_20180331.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print(', '.join(row))
        print("row : ", row)
        np.append(london_aqui_all_stations, row, axis = 0)
        print(london_aqui_all_stations)

print("test row : ")
print(london_aqui_all_stations)
'''
print(csv_reader[0])
print(csv_reader[1])

london_aqui_all_stations = np.array([ for row in csv_reader])

print("here all array : ")
print(london_aqui_all_stations)
'''
