

import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__
#after tensorflow2 keras is embeded in tensorflow



dataset = pd.read_excel('data.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# rescale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





ann = tf.keras.models.Sequential()
# tensorflow 2 includes keras in it

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1))
# can also use sigmoid or softmax(multiple outputs) activation function for classification
# but in regression we do not need 


ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
# adam is the optimizer that covers stochastic gradient descent
# adam is used for stochastic gradient descent


ann.fit(X_train, y_train, batch_size = 32, epochs = 100)



y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)# otherwise numbers are going to be huge
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# Y is horizontal, to make it vertical "reshape" is used. [len is length]
# the last "1" is for vertical concatination, if it was horizontal we should put zero.

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)