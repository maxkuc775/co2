###         CO2

# import libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Die Datei muss erst aufbereitet werden,
# Es werden erst alle Werte je Land und Jahr aufsummiert
# Es wird nur eine Zeile je Land erzeugt und summierten Werte je Jahr nach rechts weggeschrieben

rel_path = os.path.dirname(os.path.realpath(__file__))  # Pfad zum Projekt
csv_data_file = rel_path + '/greenhouse_gas_inventory_data_data.csv'  # Datei muss im Projektordner liegen
csv_output_file = rel_path + '/new.csv'

print("Input: {}".format(csv_data_file))
# Lies csv. Nur Felder: Land, Jahr, Emission
df = pd.read_csv(csv_data_file, usecols=['country_or_area', 'year', 'value'])

# Gruppiere Daten (vor 2015) nach Land und Jahr, summiere Emission
df = df[df.year < 2015].groupby(['country_or_area', 'year']).sum(level='value')

# Erstelle Pivot-Tabelle mit Land als Index, Jahre als Spalten
# https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html#pivot-tables
pivot = df.pivot_table(index='country_or_area',
                       columns='year',
                       values='value')

print("Output: {}".format(csv_output_file))
# Schreib csv
pivot.to_csv(csv_output_file)

#########################################################################
# Einlesen der neuen Datei, die so aufbereitet ist, wie wir sie brauchen
df = pd.read_csv(rel_path + '/new.csv')

# Länder in List "countries" schreiben
matrix2 = df[df.columns[0]].values
countries = matrix2.tolist()
print("Laender: ", countries)

# Jedem Land eine individuelle Zahl zuweisen und in Cluster schreiben
def converter(country):
    return countries.index(country)

df['Cluster'] = df['country_or_area'].apply(converter)

# neue CSV Datei mit dem Namen new2.csv speichern
df.to_csv('new2.csv')

# Neue Datei einlesen
# We'll set the 2nd column as the index --> Values beginnen hier (1990)
df = pd.read_csv(rel_path + '/new2.csv', index_col=3)

#print(df.head())
#print(df.info())
#print(df.describe())

### X and y arrays
# What are the columns of the dataset that you want to use as inputs (x-values) for the classification? Adapt the list columns accordingly. What is the column for the ys, i.e., the predictions?
col = list((df.columns))
col.remove('2014')
col.remove('country_or_area')
columns = col
X = df[columns]
y = df['2014']

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