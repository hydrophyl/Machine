from __future__ import print_function
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

#Training set
#Temperatur (Cecius)
X1 = np.array([[10.7, 4.9, 3.8, 1.7, 8.9, 8.9, 8.8, 13.4, 19.8, 18.9, 15.8, 12.1, 8.6, 5.8, 2.3, 3.3, 14.9, 18.8, 22.8, 27.3, 30.7, 30.6, 27.6, 23.2]])
#Niederschlage
X2 = np.array([[6.1 , 3.4, 5.9, 4.7, 2.6, 1.8, 1.9, 1.7 , 0.0 , 0.1 , 0.0 , 1.4 , 3.8, 1.6, 2.9, 5.3, 0.5 , 0.3 ,  0  ,  0.3, 0   ,  0  , 0   , 0.2]])
#Gutes oder schlechtes Wetter
y  = np.array([[1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0]])

#Building Xbar
one = np.ones()
Xbar = np.concatenate((one, X), axis = 1)

