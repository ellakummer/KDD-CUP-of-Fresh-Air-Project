import csv
import numpy as np

# transform into array London_historical_aqi_forecast_stations_20180331

column1 = np.empty((0,1))
column2 = np.empty((0,1))
column3 = np.empty((0,1))
column4 = np.empty((0,1))
column5 = np.empty((0,1))

with open('../final_project_data/London_historical_aqi_forecast_stations_20180331.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        #print(', '.join(row))
        if line_count != 0 :
            column1 = np.append(column1, np.array([[row[1]]]))
            column2 = np.append(column2, np.array([[row[2]]]))
            column3 = np.append(column3, np.array([[float(row[3])]]))
            column4 = np.append(column4, np.array([[float(row[4])]]))
            column5 = np.append(column5, np.array([[float(row[5])]]))
        line_count += 1
        #if line_count == 100000 : #test almost all
        #if line_count == 10800 : #test first id
        if line_count == 11 : #test smal
        #if line_count == 300 : #test ok for first id
            break


print("----------- TESTS ------------")

print("COMBINE : ")
records = np.rec.fromarrays((column1, column2, column3, column4, column5), names=('date', 'id', 'PM2.5', 'PM10', 'N02'))
print(records)
print("DATE : ")
print(records['date'])
print("N02 : ")
print(records['N02'])
print("PM2.5 : ")
print(records['PM2.5'])

print("#datas : ", line_count)
print("test shape : ")
print(records.shape)

print("7th element : ")
print(records[7])
print("9th element : ")
print(records[9])
print("1st parameter 9th element : ")
print(records[9][0])
print("3rd parameter 9th element : ")
print(records[9][2])
print("last element : ")
print(records[-1])
print("test sum :")
print(records[7][2] + records[9][2])

print("----------- END TEST ------------")

'''
add second datas : London_historical_aqi_other_stations_20180331
'''
