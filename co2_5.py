### CO2 Data science123

#import libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Pfad zum Projekt = Relativer Pfad
rel_path = os.path.dirname(os.path.realpath(__file__))

# Variablen der Ein- und Ausgabe - Datei
csv_data_file = rel_path + '/greenhouse_gas_inventory_data_data.csv'  # Datei muss im Projektordner liegen
csv_output_file = rel_path + '/neu.csv' #Output File

# Ausgabe: Was ist die Input - Datei
print("Input: {}".format(csv_data_file))

# Lies csv. Nur Felder: Land, Jahr, Emission
# Spalte Kategorie wird weggelassen, da nicht berücksichtigt und aufsummiert
df = pd.read_csv(csv_data_file, usecols=['country_or_area', 'year', 'value'])

#year = Spalte 1 ...
year = df[df.columns[1]].values
emissions = df[df.columns[2]].values

# Gruppiere Daten (vor 2015) nach Land und Jahr, summiere Emission der verschiedenen Kategorien
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
#print (df.head()) #gibt erste 5 Zeilen wieder
#print (df.info()) #Anzahl Spalten, RangeIndex, Datentypen, ...
#print (df.describe()) #Anzahl Zeilen, Durchschnitt, Abweichung, ...
#print (df.columns) #Namen der Spalten

# Länder in List "countries" schreiben
matrix2 = df[df.columns[0]].values #Länder werden als String in Array matrix2 geschrieben
countries = matrix2.tolist() # matrix2 wird zur Liste countries gemacht
print("Laender: ", countries)

# Jedem Land eine individuelle Zahl zuweisen und vorne vorweg schreiben
def converter(country):
    return countries.index(country)

# neue CSV Datei mit dem Namen new2.csv speichern
df.to_csv('neu2.csv')

# neue CSV Datei mit dem Namen new2.csv einlesen
df = pd.read_csv(rel_path + '/neu2.csv')

print("Beginn des linearen Regression")
### Training a Linear Regression Model
## Definiere X und Y Arrays
# Droppen von Country, 2013, 2014 und der ersten Spalte (Unnamed 0)
# String geht nicht, 2013 target variable, 2014 soll vorhergesagt werden, erste Spalte (Länderzahlen 0) irrelevant
X = df.drop(['country_or_area','2013','2014','Unnamed: 0'],1)
y = df['2013'] #target variable

### Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

### Creating and Training the Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

### Model evaluation
coeff_df = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

### Predictions from our Model
predictions = lr.predict(X_test)

### Regression Evaluation Metrics
from sklearn import metrics
### Mean Absolute Error, Mittlere absolute Abweichung
print('MAE LR: ', metrics.mean_absolute_error(y_test, predictions))
mae_lr = metrics.mean_absolute_error(y_test, predictions)

#Plotten eines Landes
xs = np.array(range(1990,2015))
for country in countries[1:2]:
    row = df.loc[df["country_or_area"] == country]
    ys = np.array(row._values[0][2:], dtype=float)
    ys_lr = ys.copy()
    #ys_lr[-1] = predictions
    sns.lineplot(xs, ys)
    sns.lineplot(xs, ys_lr)
    plt.show()
    #prediction 2014 eintragen und regressionsgerade
    #danach neuronales netz und vergleichen
    #1990 - 2012 oder Länder aufteilen in Testdaten

############################################################
print(" ")
print("Beginn des Neuronalen Netzes")

dataset = pd.read_csv(rel_path + '/neu2.csv')
dataset = dataset.drop('country_or_area',1)
dataset = dataset.drop('Unnamed: 0',1)
#dataset.to_csv('neu3.csv')
#dataset = pd.read_csv(rel_path + '/neu3.csv')

#normieren der Daten
"""
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
"""

def build_model():
  input_len = len(X_train.columns)
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[input_len]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.0001)
  optimizer = tf.keras.optimizers.Adam()

  model.compile(loss='mse', # We optimize for minimizing the mean squared error (mse)
                optimizer=optimizer,
                metrics=['mae', 'mse']) # We evaluate with the mean absolute error (mae) and the mean squared error (mse)
  return model

model = build_model()
print(model.summary())

example_batch = X_train[:10]
example_result = model.predict(example_batch)
#print(example_result)

##### TRAIN THE MODEL ###################
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

# EPOCHS
EPOCHS = 300

history = model.fit(
  X_train, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.head())
hist.to_csv('neu4.csv')

#Diagramm
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  #plt.ylim([0,5])
  plt.legend()
  plt.show()

plot_history(history)

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)
plt.show()

loss, mae, mse = model.evaluate(X_test, y_test, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} ".format(mae))
mae_nn=mae

test_predictions = model.predict(X_test).flatten()

plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [CO2]')
plt.ylabel('Predictions [CO2]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - y_test
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [CO2]")
plt.ylabel("Count")
plt.show()

print('MAE LR: ',mae_lr)
print('MAE NN: ',mae_nn)

datenbeschriftung =['MAE LR','MAE NN']
data = [mae_lr, mae_nn]
s = pd.Series(data, index=datenbeschriftung)
s.plot(kind="bar", rot=0)
plt.xlabel('MAE')
plt.plot()
plt.show()
#test