import csv
import numpy as np

# transform into array

london_aqui_all_stations = np.empty((0,5))


with open('../final_project_data/London_historical_aqi_forecast_stations_20180331.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 10 : #test
            break
        #print(', '.join(row))
        london_aqui_all_stations = np.append(london_aqui_all_stations, np.array([row[1:6]]), axis = 0)
        line_count += 1

print("----------- END TEST ------------")
print("test row : ")
print(london_aqui_all_stations.shape)
print(london_aqui_all_stations)
print("test select : ", london_aqui_all_stations[9])
print("test select : ", london_aqui_all_stations[9][0])
