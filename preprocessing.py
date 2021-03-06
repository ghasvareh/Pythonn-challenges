# data preprocessing

# importing libraries
import numpy as np # does math on dataset
import matplotlib.pyplot as plt  # graph
import pandas as pd # import dataset
# importing datasets
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values #independent variables (iloc is used when it is not numeric or we are not sure about the format of the  variable)

#dependent variables
Y=dataset.iloc[:,3].values # the index starts from zero
#taking care of missing data (replacing with the averages in each column)
from sklearn.preprocessing import Imputer #imputer with capital "I"
imputer=Imputer (missing_values='NaN',strategy='mean',axis=0) #with capital N in 'NaN'
imputer=imputer.fit(X[:,1:3]) #fit imputer  "only" to the columns that have missing data
# 1:3 because the upper bound is excluded
X[:,1:3]=imputer.transform(X[:,1:3]) # replace the missing data



##### encoding categoricla Data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder() ##function for specific matrix
X[:,0]=labelencoder_x.fit_transform(X[:,0])
onehotencoder = OneHotEncoder (categorical_features = [0]) # we have multiple culumns and is non-comparable
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder() #what does that do?
#Y=labelencoder_y.fit_transform(X[:,0]) # they are dependant and are not comparable by definition
Y=labelencoder_y.fit_transform(Y)

#####
# splitting the data in to training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2) # ""random_state=0"" is good for teaching since we have same results
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train) # dont need to fit for the test set
X_test=sc_X.transform(X_test) #we dont rescale Y since it is categorical quantity