# probability of going having bad credit

# Installing Keras
# pip install --upgrade keras (Theano and Tensorflow also needs to be installed)

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # to avoid dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) #random_state = 0 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#  making the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential #to initialize nural network
from keras.layers import Dense #to build layers

# Initialising the ANN
classifier = Sequential() #classification problem
# model = Sequential() if it is regression problem

# Adding the input layer and the first hidden layer #
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#  "11" comes from dimention of "X"
# "relu" stands for rectifying fcn which is used for hidden layers
# "6" stands for nodes in the first hidden layer
# 'uniform' stands for iniform distribution for initializing the weights

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# "model.add" if it is regression problem 


classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# softmax if we have three or more categories

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# 'adam' refers to the algorithm that makes more efficient [type of stochastic gradient descent]
# 'loss' gradient descent is based on loss function 'logarithmic loss' for binaaary. for thrre options we have 'categorical_crossentropy'
# 'mean_squared_error' is a 'loss' for regression 
# 'accuracy' is usually the case to imrove the model
# in 'metrics' '[]' can have multiple metrics

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# 'batch_size' number of observation after which you want to update the weights
# 'nb_epoch'  how many round do we go to optimiza

# Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #probabiility over 50% (need it for confusion matrix)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# accuracy is coorect guess devided by total observations in the test set
