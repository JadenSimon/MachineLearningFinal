import process_data
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier as bagging
from sklearn.ensemble import AdaBoostClassifier as adaboost
from sklearn.neural_network import MLPClassifier as mlp

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

dataset_train = np.array(converted[:7000], dtype='float64')  # 7000 samples for training
dataset_test = np.array(converted[7000:], dtype='float64')  # 601 samples for testing

dataset_train_x = preprocessing.scale(dataset_train[:, :-1])
dataset_train_y = dataset_train[:, -1]
dataset_test_x = preprocessing.scale(dataset_test[:, :-1])
dataset_test_y = dataset_test[:, -1]

# SVM
clf = SVC(C=0.75, gamma=2.0)
clf.fit(dataset_train_x, dataset_train_y)
print("SVM Training Score: {}".format(round(clf.score(dataset_train_x, dataset_train_y), 2)))
print("SVM Testing Score: {}".format(round(clf.score(dataset_test_x, dataset_test_y), 2)))

# Bagging with decesion stumps
clf = bagging(n_estimators=200, oob_score=True)
clf.fit(dataset_train_x, dataset_train_y)
print("Bagging Training Score: {}".format(round(clf.score(dataset_train_x, dataset_train_y), 2)))
print("Bagging Testing Score: {}".format(round(clf.score(dataset_test_x, dataset_test_y), 2)))

# Adaboost
clf = adaboost(n_estimators=50, learning_rate=.3)
clf.fit(dataset_train_x, dataset_train_y)
print("Adaboost Training Score: {}".format(round(clf.score(dataset_train_x, dataset_train_y), 2)))
print("Adaboost Testing Score: {}".format(round(clf.score(dataset_test_x, dataset_test_y), 2)))

# Multi-level Perceptron Neural Network
clf = mlp(activation='relu', alpha=1e-05, batch_size='auto',
          beta_1=0.9, beta_2=0.999, early_stopping=False,
          epsilon=1e-08, hidden_layer_sizes=(5, 2),
          learning_rate='constant', learning_rate_init=0.001,
          max_iter=200, momentum=0.9,
          nesterovs_momentum=True, power_t=0.5, random_state=1,
          shuffle=True, solver='lbfgs', tol=0.0001,
          validation_fraction=0.1, verbose=False, warm_start=False)
clf.fit(dataset_train_x, dataset_train_y)
print("Neural Network Training Score: {}".format(round(clf.score(dataset_train_x, dataset_train_y), 2)))
print("Neural Network Testing Score: {}".format(round(clf.score(dataset_test_x, dataset_test_y), 2)))


# best_test_list = []
# best_train_list = []
# for g in np.arange(.05, .5, .05):
#     for c in range(50, 500, 50):
#         clf = adaboost(n_estimators=c, learning_rate=g)
#         clf.fit(dataset_train_x, dataset_train_y)
#         best_train_list.append((c, g, str(clf.score(dataset_train_x, dataset_train_y))))
#         best_test_list.append((c, g, str(clf.score(dataset_test_x, dataset_test_y))))

# print(max(best_test_list, key=lambda iter: iter[2]))
# print(max(best_train_list, key=lambda iter: iter[2]))
