#CO2

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in Data. We'll set the first column as the index.
df = pd.read_csv("F:\Studium\Master\Python\greenhouse_gas_1country.csv")

year = df.year #Jahre einlesen in year
value = df.value #Werte einlesen value
arr1 = np.array([value, year]) #List als Array
f_arr1 = np.flip(arr1) #Array flippen, nur für bessere Darstellung

a = 0 # für die for-Schleife
for i in range(1990, 2015): # für die Jahre 1990 bis 2014
    df[i] = f_arr1[1,a]
    a = a+1 # der nächste Array-Wert


""" ----- Vorbereitung für die Regressionsanalyse ----- """
cols_ratio = ['1990','1991','1992']
cols_target = ['2014']

dataSet_ratio = dataSet_c.loc[:, cols_ratio]
dataSet_target = dataSet_c[cols_target]

X = dataSet_ratio[['1992']]  # doppelte [], da eine Liste von Spalten zu übergeben ist
y = dataSet_c[cols_target]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.3,  # 70% der Daten für das Training
    random_state=None)  # bei Bedarf kann hier "dem Zufall auf die Sprünge geholfen" werden

""" ----- Lineare Regressionsanalyse ------- """

from sklearn.linear_model import LinearRegression  # importieren der Klasse

lr = LinearRegression()  # instanziieren der Klasse

lr.fit(X_train, y_train)  # trainieren

print('------ Lineare Regression -----')
print('Funktion via sklearn: y = %.3f * x + %.3f' % (lr.coef_[0], lr.intercept_))
print("Alpha: {}".format(lr.intercept_))
print("Beta: {}".format(lr.coef_[0]))
print("Training Set R² Score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test Set R² Score: {:.2f}".format(lr.score(X_test, y_test)))
print("\n")