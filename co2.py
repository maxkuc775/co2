#CO2

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in Data. We'll set the first column as the index.
df = pd.read_csv("F:\Studium\Master\Python\greenhouse_gas_1country.csv", index_col=0)

# Kopf ausgeben
#print(df.head())

year = df.year #Jahre einlesen in year
value = df.value #Werte einlesen value
arr1 = np.array([value, year]) #List als Array
f_arr1 = np.flip(arr1) #Array flippen, nur für bessere Darstellung

#a = 24 # für die for-Schleife
#for i in range(2014, 2015): # für die Jahre 1990 bis 2014
#    df[i] = f_arr1[1,a]
#    a = a+1 # der nächste Array-Wert
#####################################################



### X and y arrays
# What are the columns of the dataset that you want to use as inputs (x-values) for the classification? Adapt the list columns accordingly. What is the column for the ys, i.e., the predictions?

df['Cluster'] = f_arr1[1,24]

col = list((df.columns))
col.remove('category')
col.remove('Cluster')
columns = col
X = df[columns]
y = df['Cluster']

# Train Test Split
from sklearn.model_selection import train_test_split

# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

## Create and train the classifier
# Import the KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Instantiate the classifier with 3 neighbors
neigh = KNeighborsClassifier(n_neighbors=3)

# fit the KNeighbors classifier to the dataset
neigh.fit(X, y)
print("Abc", neigh.predict(X_test))
#print(neigh.predict_proba(X_test))

## Model Evaluation
# The accuracy can be obtained with the score function of the classifier.
correct2 = y_test == np.array(neigh.predict(X_test))
accuracy2 = np.mean(correct2)
print(accuracy2)
