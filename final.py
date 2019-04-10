import process_data
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import preprocessing

# Some test stuff
_, epa = process_data.load_dataset('epa_data.csv')
_, meso = process_data.load_dataset('meso_data.csv')

# Create a dataset
dataset = []

for date, epa_data in epa.items():
	if date in meso:
		meso_data = meso[date]
		category = epa_data[1]

		if category == 'Hazardous' or category == 'Very Unhealthy':
			category = 'Unhealthy'

		dataset.append(meso_data + [category]) 

names, converted = process_data.convert_class(dataset)
converted = process_data.time_series(converted)

dataset_train = np.array(converted[:7000], dtype='float64') # 7000 samples for training
dataset_test = np.array(converted[7000:], dtype='float64') # 601 samples for testing

dataset_train_x = preprocessing.scale(dataset_train[:,:-1])
dataset_train_y = dataset_train[:,-1]
dataset_test_x = preprocessing.scale(dataset_test[:,:-1])
dataset_test_y = dataset_test[:,-1]

clf = SVC(C=0.75, gamma=2.0)
clf.fit(dataset_train_x, dataset_train_y)

print("Training Score: " + str(clf.score(dataset_train_x, dataset_train_y)))
print("Testing Score: " + str(clf.score(dataset_test_x, dataset_test_y)))

