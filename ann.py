# Import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# import dataset
data = pd.read_csv('/home/dwijendra/Desktop/Customer-Churn/Churn_Modelling.csv')
data.head()

X = data.iloc[:,3:-1]
y = data.iloc[:,-1]

# create dummy variables
geography = pd.get_dummies(data = X['Geography'],drop_first=True)
gender = pd.get_dummies(data = X['Gender'],drop_first=True)

# concat two new columns to the X dataframe
X = pd.concat([X,geography,gender],axis=1)

# drop unnecessary columns
X = X.drop(['Geography','Gender'],axis=1)

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Building ANN
## initializing ann
classifier = Sequential()

## adding 1st hidden layer
classifier.add(Dense(units=6,activation='relu',kernel_initializer='he_uniform',input_dim = 11))

##  adding 2nd hidden layer
classifier.add(Dense(units=6,activation='relu',kernel_initializer='he_uniform'))

## adding output layer
classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='glorot_uniform'))

## compiling ann
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Train ANN
# training ann on the train set
classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=100)

# predicting test set results
y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)

# make confusion_matrix and get accuracy score
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)