### CO2

#import libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pfad zum Projekt = Relativer Pfad
rel_path = os.path.dirname(os.path.realpath(__file__))

# Variablen der Ein- und Ausgabe - Datei
csv_data_file = rel_path + '/greenhouse_gas_inventory_data_data.csv'  # Datei muss im Projektordner liegen
csv_output_file = rel_path + '/neu.csv' #Output File

# Ausgabe: Was ist die Input - Datei
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

# Ausgabe: Was ist die Ausgabe - Datei
print("Output: {}".format(csv_output_file))

# Schreib csv (erstelle neue CSV Datei)
pivot.to_csv(csv_output_file)

# Einlesen der neuen Datei, die so aufbereitet ist, wie wir sie brauchen
df = pd.read_csv(rel_path + '/neu.csv')

# Ein paar Infos der Datei anzeigen
print (df.head()) #gibt erste 5 Zeilen wieder
print (df.info()) #Anzahl Spalten, RangeIndex, Datentypen, ...
print (df.describe()) #Anzahl Zeilen, Durchschnitt, Abweichung, ...
print (df.columns) #Namen der Spalten

# Länder in List "countries" schreiben
matrix2 = df[df.columns[0]].values
countries = matrix2.tolist()
print("Laender: ", countries)

# Jedem Land eine individuelle Zahl zuweisen und vorne vorweg schreiben
def converter(country):
    return countries.index(country)

# neue CSV Datei mit dem Namen new2.csv speichern
df.to_csv('neu2.csv')

# neue CSV Datei mit dem Namen new2.csv einlesen
df = pd.read_csv(rel_path + '/neu2.csv')
print (df.head())

# Ein paar simple Diagramme erstellen
#sns.pairplot(df) # Erstellen der Diagramme
#sns.distplot(df['year'])
#plt.show() # Anzeigen der Diagramme

### Training a Linear Regression Model
## Definiere X und Y Arrays
# Droppen von Country, da String und 2014, da target variable
X = df.drop(['country_or_area','2014'],1)
y = df['2014'] #target variable

### Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

### Creating and Training the Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

### Model evaluation
coeff_df = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

### Predictions from our Model
predictions = lr.predict(X_test)
plt.scatter(y_test, predictions)
plt.show()

### Regression Evaluation Metrics
from sklearn import metrics
### Mean Absolute Error
print('MAE: ', metrics.mean_absolute_error(y_test, predictions))