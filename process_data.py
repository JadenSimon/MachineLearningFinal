# This script processes all raw datasets into a more compact form, removing all unnecessary features.
# Also contains functions to create new time-series datasets.

# EPA data: https://aqs.epa.gov/aqsweb/airdata/download_files.html#AQI
# Meswest data: https://mesowest.utah.edu/
# NOAA data: https://www.ncdc.noaa.gov/cdo-web/search

# EPA data is originally in the form [State, County, State Code, County Code, Date, AQI, Category, Defining Parameter, Defining Site, # of Sites]
# We want to focus on Salt Lake county so we will remove all other data. Also need to remove unneeded features.

# Meso data units:
# Temperature - C
# Wind_X, Wind_Y - m/s
# Humidity - %
# Pressure - Pascals

import csv
import math
import dateutil.parser as parser
from datetime import timedelta
from os import listdir
from operator import add

# Checks if the input exists, if not returns 0, otherwise converts it to a float
def verify_input(data):
	if data == '':
		return 0.0
	else:
		return float(data)

# Processes a AQI data file from the EPA
def process_epa_data(file_name):
	data = {}

	with open(file_name) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')

		# Look for rows with Utah, Salt Lake
		for row in csv_reader:
			if row[0] == 'Utah' and row[1] == 'Salt Lake':
				data[row[4]] = [row[5], row[6], row[7]]

	return data

# Processes all files in raw_epa_data folder, creating a single epa_data csv file
def create_epa_dataset():
	with open('epa_data.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(['Date', 'AQI', 'Category', 'Parameter']) # Adds a header

		# Iterate over all datasets
		for f in listdir('raw_epa_data'):
			processed_data = process_epa_data('raw_epa_data/' + f)
			for date, data in processed_data.items():
				writer.writerow([date] + data)

# Processes weather data from MesoWest
# Some of the data is useless, while some of it is recorded every hour
# Thus we need to process the data into a more useful format, such as avg. temp for the day, avg. wind speed, etc.
# The first index will be the date in the format 'YYYY-MM-DD' to match the EPA data
# Current output is [Date] = [Temp (C), Wind (m/s), Humidity (%), Pressure (P)]
def process_meso_data(file_name):
	processed_data = {}

	with open(file_name) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')

		# Get the CSV headers and units
		header = next(csv_reader)
		units = next(csv_reader)

		# Determine indices to access
		temp_index = header.index('air_temp_set_1')
		humidity_index = header.index('relative_humidity_set_1')
		speed_index = header.index('wind_speed_set_1')
		angle_index = header.index('wind_direction_set_1')
		pressure_index = header.index('pressure_set_1d')

		# Initialize stuff on the first line
		# We need to group data by the day, so keep a running count of all daily averages we would like to track
		first_line = next(csv_reader)
		current_date = first_line[1].split('T', 2)[0]
		avg_temp = verify_input(first_line[temp_index])
		avg_humidity = verify_input(first_line[humidity_index])
		speed = verify_input(first_line[speed_index])
		angle = verify_input(first_line[angle_index])
		avg_wind = [speed * math.cos(math.radians(angle)), speed * math.sin(math.radians(angle))]
		avg_pressure = verify_input(first_line[pressure_index])
		count = 1.0

		for row in csv_reader:
			date, time = row[1].split('T', 2) # Get the date from the timestamp

			# End of sequence, store averages
			if date != current_date:
				processed_data[current_date] = [avg_temp/count, avg_wind[0]/count, avg_wind[1]/count, avg_humidity/count, avg_pressure/count]
				avg_temp = 0.0
				avg_wind = [0.0, 0.0]
				avg_humidity = 0.0
				avg_pressure = 0.0
				count = 0.0
				current_date = date

			# Update averages
			avg_temp += verify_input(row[temp_index])
			avg_humidity += verify_input(row[humidity_index])
			speed = verify_input(row[speed_index])
			angle = verify_input(row[angle_index])
			avg_wind[0] += speed * math.cos(math.radians(angle))
			avg_wind[1] += speed * math.sin(math.radians(angle))
			avg_pressure += verify_input(row[pressure_index])
			count += 1

		# Dump the last value
		processed_data[current_date] = [avg_temp/count, avg_wind[0]/count, avg_wind[1]/count, avg_humidity/count, avg_pressure/count]

	return processed_data

# Processes all raw data from MesoWest into a single CSV file
# Raw data that shares the same date will be merged and averaged
def create_meso_dataset():
	datasets = []
	processed_data = {}

	# First collect all the datasets to be merged
	for f in listdir('raw_meso_data'):
		datasets.append(process_meso_data('raw_meso_data/' + f))

	# Now combine them
	merge_count = {}
	for dataset in datasets:
		for date, data in dataset.items():
			if date in merge_count:
				processed_data[date] = list(map(add, processed_data[date], data))
				merge_count[date] += 1
			else:
				merge_count[date] = 1
				processed_data[date] = data

	# Divide each element by the mergecount to average them
	for date, data in processed_data.items():
		processed_data[date] = [x / merge_count[date] for x in data]

	# Write the merged data to a csv file
	with open('meso_data.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(['Date', 'Temperature', 'Wind_X', 'Wind_Y', 'Humidity', 'Pressure']) # Adds a header

		for date, data in processed_data.items():
			writer.writerow([date] + data)

	return processed_data

# Loads a dataset, returning a list containing the headers and a dictionary containing the data
def load_dataset(file_name):
	header = []
	data = {}

	with open(file_name) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		header = next(csv_reader)
		header.pop(0) # Gets read of the date part

		for row in csv_reader:
			data[row[0]] = row[1:]

	return header, data

# Creates a time series model from a dataset
# Assumes last index of dataset is the label
def time_series(dataset):
	new_dataset = []

	# For every element except the last, relabel the example using the next example's label
	for i in range(len(dataset) - 1):
		new_dataset.append(dataset[i] + [dataset[i+1][-1]])

	return new_dataset

# Converts a categorical label into a set of classes
def convert_class(dataset):
	converted_dataset = list(dataset)
	classes = {}
	class_count = 0

	for i in range(len(dataset)):
		class_label = dataset[i][-1]

		# Found a new label
		if class_label not in classes:
			class_count += 1
			classes[class_label] = class_count

		converted_dataset[i][-1] = classes[class_label]

	return classes, converted_dataset
