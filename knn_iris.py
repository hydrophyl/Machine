import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
#load datasets and show somes datas
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('Number of classes: %d' %len(np.unique(iris_y)))
print('Number of data points: %d' %len(iris_y))

X0 = iris_X[iris_y == 0,:]
print('\nSamples from classe 0:\n', X0[:5,:])

X1 = iris_X[iris_y == 1,:]
print('\nSamples from classe 1:\n', X1[:5,:])

X2 = iris_X[iris_y == 2,:]
print('\nSamples from classe 0:\n', X2[:5,:])

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size =50)

print("Training size: %d" %len(y_train))
print("Test size: %d" %len(y_test))

def myweight(distance):
    sigma2 = .5 #we can change this number
    return np.exp(-distance**2/sigma2)

#K = 1 -> moi diem test data, chon 1 diem training data gan nhat va gan label cua data do cho test_data
clf = neighbors.KNeighborsClassifier(n_neighbors= 10, p = 2, weights= myweight)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#print("Print results  for 20 test data points: ")
#print("Predicted labels: ", y_pred[20:40])
#print("Ground truth:     ", y_test[20:40])

print("Accuracy of 10NN (1/distance weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))