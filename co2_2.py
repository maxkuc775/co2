###         CO2 Vorhersage

# import libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Die Datei muss zuerst aufbereitet werden,
# Es werden erst alle Werte je Land und Jahr aufsummiert
# Es wird nur eine Zeile je Land erzeugt und summierten Werte je Jahr nach rechts weggeschrieben
# (Monaco entfernt, da keine Werte für 2014)

########################################################

#Vorbereiten der Input Datei
rel_path = os.path.dirname(os.path.realpath(__file__))  # Pfad zum Projekt
csv_data_file = rel_path + '/greenhouse_gas_inventory_data_data.csv'  # Datei muss im Projektordner liegen
csv_output_file = rel_path + '/new.csv' #Output File

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

# Jedem Land eine individuelle Zahl zuweisen
def converter(country):
    return countries.index(country)

#neue Spalte CLuster mit individueller Zahl
#df['Cluster'] = df['country_or_area'].apply(converter)

# neue CSV Datei mit dem Namen new2.csv speichern
df.to_csv('new2.csv')

# Neue Datei einlesen
# We'll set the 2nd column as the index --> Values beginnen hier (1990)
df = pd.read_csv(rel_path + '/new2.csv')

#print(df.head())
#print(df.info())
#print(df.describe())
#print(df.dtypes)
#print(df.tail(11))

#features = Werte aus Vergangenheit, die benutzt werden, um 2014 rauszufinden
#daher 2014 raus
#country_or_area ist string, macht probleme, aber jedes country hat eine zahl
features = np.array(df.drop(['country_or_area','2014'],1))
# Label = was wir vorhersagen wollen
labels = np.array(df.filter(items=['2014']))

#vorbereiten der features
#jedes feature wird skaliert
features = preprocessing.scale(features)

#aufteilen in testdaten und 20% trainingsdaten
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

#Modell wird trainiert
linear_classifier = LinearRegression()
linear_classifier.fit(features_train, labels_train)

#Überprüfen, wie gut die Daten sind
score = linear_classifier.score(features_test, labels_test)
print('Genauigkeit: ', score)

######################################################
# Hier nur für den Germany Plot
# Relativer Pfad der Datei (im Projektordner)
rel_path = os.path.dirname(os.path.realpath(__file__))
#Input File = Original Datei
csv_data_file = rel_path + '/greenhouse_gas_inventory_data_data.csv'
#Output File - Datei wird angepasst zum Bearbeiten
csv_output_file = rel_path + '/new3.csv'

#Ausgabe was Input File ist
#print("Input: {}".format(csv_data_file))
# Lies nur Felder: Land, Jahr, Emission aus Input File
df2 = pd.read_csv(csv_data_file, usecols=['country_or_area', 'year', 'value'])

# Gruppiere Daten nach Land und Jahr, summiere Emission
df2 = df2.groupby(['country_or_area', 'year']).sum(level='value')

#Ausgabe was Output File ist und Speichern neue Datei
#print("Output: {}".format(csv_output_file))
df2.to_csv(csv_output_file)

df2 = pd.read_csv(rel_path + '/new3.csv')
#Nur Deutschland
df2 = df2.loc[(df2['country_or_area'] == 'Germany')]
df2.to_csv('new3.csv')

#Formatieren des Datums
df2.loc[:,'year'] = pd.to_datetime(df2.year, format= '%Y')
df2.set_index('year', inplace = True)

#print(df2.head())
#Diagramm machen
ax = plt.gca()
df2.plot(kind='line',y='value',ax=ax)
plt.show()

