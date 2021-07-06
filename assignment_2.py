# -*- coding: utf-8 -*-
"""Assignment-2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r0wemTUbrHZd_PvXDYMclpTB1nqaj6d-

# Artificial Neural Network

### Importing the libraries
"""

import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

"""## Part 1 - Data Preprocessing

### Importing the dataset
"""

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)

print(y)

dataset.head(10)

"""### Encoding categorical data

Label Encoding the "Gender" column
"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

print(X)

"""One Hot Encoding the "Geography" column"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

print(X[0])

dataset.info()

"""### Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""### Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train[0])

"""## Part 2 - Building the ANN

### Initializing the ANN
"""

ann = tf.keras.models.Sequential()

"""### Adding the input layer and the first hidden layer"""

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

"""### Adding the second hidden layer"""

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

"""### Adding the output layer"""

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"""## Part 3 - Training the ANN

### Compiling the ANN
"""

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""### Training the ANN on the Training set"""

#train on 100,500,1000 epochs
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

"""## Part 4 - Making the predictions and evaluating the model

### Predicting the result of 3 observation

Therefore, our ANN model predicts that this customer stays in the bank!

**Important note 1:** Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.

**Important note 2:** Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.For Male you should encode as '1' and for Female '0'

**Assignment**

Use our ANN model to predict if the customers with the following informations will leave the bank: 

Geography: France,Germany,Spain

Credit Score: 600,800,700

Gender: Male,Female,Male

Age: 40,50,35,years old

Tenure: 3,5,4 years

Balance: \$ 60000,70000,0

Number of Products: 2,1,0

Does this customer have a credit card ? Yes,No,No

Is this customer an Active Member: Yes,No,No

Estimated Salary: \$ 50000,10000,0

So, should we say goodbye to that customer ?

Sample 
If value is greater than 0.5 - True  
Else -False
"""

print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]])) > 0.5)

ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))

print(ann.predict(sc.transform([[0,1,0,800,0,50,5,70000,1,0,0,10000]])) > 0.5)

print(ann.predict(sc.transform([[0,0,1,700,1,35,4,0,0,0,0,0]])) > 0.5)



"""### Predicting the Test set results"""

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

y_pred[0:10]

len(y_test)

"""### Making the Confusion Matrix"""

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy_score(y_test, y_pred)

import matplotlib.pyplot as plt
import seaborn as sns
LABELS = ['Not Left Bank', 'Left Bank'] 
plt.figure(figsize =(8, 8)) 
sns.heatmap(cm, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show()
