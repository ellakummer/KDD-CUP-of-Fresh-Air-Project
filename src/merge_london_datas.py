import csv
import numpy as np

# transform into array London_historical_aqi_forecast_stations_20180331

london_aqui_all_stations = np.empty((0,5))

with open('../final_project_data/London_historical_aqi_forecast_stations_20180331.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        #print(', '.join(row))
        if line_count == 0:
            london_aqui_all_stations = np.append(london_aqui_all_stations, np.array([row[1:6]]), axis = 0)
        else :
            london_aqui_all_stations = np.append(london_aqui_all_stations, np.array([[row[1], row[2], float(row[3]), float(row[4]), float(row[5])]]), axis = 0)
        line_count += 1
        #if line_count == 100000 : #test
        #if line_count == 10800 : #test
        if line_count == 10 : #test
            break



print("----------- TEST ------------")

print("#datas : ", line_count)
print("test shape : ")
print(london_aqui_all_stations.shape)

print(london_aqui_all_stations)
print("8th element : ")
print(london_aqui_all_stations[7])
print("9th element : ")
print(london_aqui_all_stations[9])
print("1st parameter 9th element : ")
print(london_aqui_all_stations[9][0])
print("last element : ")
print(london_aqui_all_stations[-1])
print("test sum :")
#print(int(london_aqui_all_stations[8][2]) + int(london_aqui_all_stations[9][2])) # PASSER EN INT :!!
print(london_aqui_all_stations[7][2] + london_aqui_all_stations[9][2]) # PASSER EN INT :!!


print("----------- END TEST ------------")

# add second datas : London_historical_aqi_other_stations_20180331
'''
with open('../final_project_data/London_historical_aqi_other_stations_20180331.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 10 : #test
            break
        #print(', '.join(row))
        london_aqui_all_stations = np.append(london_aqui_all_stations, np.array([[row[1], row[0], row[2], row[3], row[4]]]), axis = 0)
        line_count += 1
        if line_count == 100000 : #test, change
            break

print("----------- TEST ------------")
print("test shape : ")
print(london_aqui_all_stations.shape)
print("----------- END TEST ------------")
'''
