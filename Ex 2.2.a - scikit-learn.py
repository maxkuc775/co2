# Exercise 2.2.a.: Comparing unsupervised clustering with supervised classification

# For this project we will K-Nearest-Neighbor classification to classify Universities into to two groups, Private and Public.
# This is the supervised learning version of the task from the lecture.
# The following lines are identical to the code discussed in the lecture:

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in the College_Data file using read_csv. We'll set the first column as the index.

df = pd.read_csv("../data/College_Data.csv",index_col=0)

print(df.head())

print(df.info())

print(df.describe())

from sklearn.cluster import KMeans

#Create an instance of a K Means model with 2 clusters.

kmeans = KMeans(n_clusters=2)

# Fit the model to all the data except for the Private label.

kmeans.fit(df.drop('Private',axis=1))

# What are the cluster center vectors?

print("Cluster centers")
print(kmeans.cluster_centers_)

# There is no perfect way to evaluate clustering if you don't have the labels, however since this is just an exercise, we do have the labels, so we take advantage of this to evaluate our clusters, keep in mind, you usually won't have this luxury in the real world.
# Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.

def converter(private):
    return int(private == "Yes")

df['Cluster'] = df['Private'].apply(converter) # aus yes und no 1 und 0 machen

print(df.head())

correct = kmeans.labels_ == np.array(df['Cluster'])

accuracy = np.mean(correct)

print("Genauigkeit: ", accuracy)

# Now implement the supervised classification:
unis = pd.read_csv('../data/College_Data.csv')
sns.distplot(df['Cluster']) #Diagramm
plt.show()

### X and y arrays
# What are the columns of the dataset that you want to use as inputs (x-values) for the classification? Adapt the list columns accordingly. What is the column for the ys, i.e., the predictions?
col = list((df.columns))
col.remove('Private')
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
