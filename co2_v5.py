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

""" ------- Vorbereitung der Daten ------ """
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
print ("Infos der Datei: ")
print (df.head()) #gibt erste 5 Zeilen wieder
print (df.info()) #Anzahl Spalten, RangeIndex, Datentypen, ...
print (df.describe()) #Anzahl Zeilen, Durchschnitt, Abweichung, ...
print (df.columns) #Namen der Spalten

#Berechnung des Mittelwerts des Jahres 2013
mittelwert = df['2013'].mean()

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

""" ------- Lineare Regression ------ """
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

""" ------- #Plotten eines Landes (Deutschland) ------ """
xs = np.array(range(1990,2015))
for country in countries[1:2]:
    row = df.loc[df["country_or_area"] == 'Germany']
    ys = np.array(row._values[0][2:], dtype=float)
    ys_lr = ys.copy()
    #ys_lr[-1] = predictions
    plt.title('Germany')
    sns.lineplot(xs, ys)
    sns.lineplot(xs, ys_lr)
    plt.show()

""" ------- Neuronales Netz ------ """
print(" ")
print("Beginn des Neuronalen Netzes")

""" --Build the model--- """
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
example_batch = X_train[:10]
example_result = model.predict(example_batch)

""" --TRAIN THE MODEL--- """
""" --Display training progress by printing a single dot for each completed epoch--- """
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

# EPOCHS
EPOCHS = 300

""" ----Train the model für 300 epochs and record the training and validation accuracy in the history object--- """
history = model.fit(
  X_train, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

""" ------- Visualize the model's training progress using the stats stored in history ------ """
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.head())
hist.to_csv('neu4.csv')

""" ------- Plot über alle Epochen ------ """
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

""" ------- Early Stop Funktion (Patience = 20!!!) und Plot ------ """
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
plot_history(history)
plt.show()

""" ------- Berechnung MAE NN ------ """
loss, mae, mse = model.evaluate(X_test, y_test, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} ".format(mae))
mae_nn=mae
test_predictions = model.predict(X_test).flatten()

""" ------- Plot um Fehler ggü. wahre Werte anzuzeigen ------ """
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [CO2]')
plt.ylabel('Predictions [CO2]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

""" ------- Plot um Verteilung der Fehler anzuzeigen ------ """
error = test_predictions - y_test
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [CO2]")
plt.ylabel("Count")
plt.show()

""" ------- Decision Tree Regressor ------ """
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X_train, y_train)
tree_predictions = tree.predict(X_test)
mae_tree = metrics.mean_absolute_error(y_test, tree_predictions)
print ('MAE TREE: ',mae_tree)

""" ----- Polynomial Regressionsanalyse ------- """
from sklearn.preprocessing import PolynomialFeatures
quadratic = PolynomialFeatures(degree = 1)
X_train_quad = quadratic.fit_transform(X_train)
X_test_quad = quadratic.fit_transform(X_test)
pr = LinearRegression()
pr.fit(X_train_quad, y_train)
pr_predictions = pr.predict(X_test_quad)
mae_pr = metrics.mean_absolute_error(y_test, pr_predictions)
print ('MAE PR: ', mae_pr)

""" ------- Linear Regressor (2) [für mehrere Durchläufe] ------ """
#Deklarieren des Numpy Arrays
list_mae_lr2 = np.array(range(0,10))
list_mae_nn2 = np.array(range(0,10))

for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101+i)
    lr2 = LinearRegression()
    lr2.fit(X_train, y_train)
    lr2_predictions = lr.predict(X_test)
    list_mae_lr2[i] = metrics.mean_absolute_error(y_test, lr2_predictions)
    mae_lr2 = list_mae_lr2.mean()

    """ ------- Berechnung MAE NN (2) [für mehrere Durchläufe] ------ """
    """ --Build the model--- """
    def build_model():
        input_len = len(X_train.columns)
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[input_len]),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.RMSprop(0.0001)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(loss='mse',  # We optimize for minimizing the mean squared error (mse)
                      optimizer=optimizer,
                      metrics=['mae',
                               'mse'])  # We evaluate with the mean absolute error (mae) and the mean squared error (mse)
        return model

    model = build_model()
    example_batch = X_train[:10]
    example_result = model.predict(example_batch)

    """ --TRAIN THE MODEL--- """
    """ --Display training progress by printing a single dot for each completed epoch--- """
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            #print('.', end='')

    # EPOCHS
    EPOCHS = 300

    """ ----Train the model für 300 epochs and record the training and validation accuracy in the history object--- """
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[PrintDot()])

    model = build_model()

    """ ------- Early Stop Funktion (Patience = 20!!!) und Plot ------ """
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

    """ ------- Berechnung MAE NN ------ """
    loss, mae, mse = model.evaluate(X_test, y_test, verbose=2)
    list_mae_nn2[i] = mae
    mae_nn2 = list_mae_nn2.mean()

print(' ')

""" ----- Ausgabe und Vergleich der Fehler der Methoden ------- """
rel_mae_lr = mae_lr2/mittelwert
rel_mae_nn = mae_nn2/mittelwert
print('Ergebnis')
print('Durchschnittlicher MAE LR aus 10 Durchläufen: ', mae_lr2)
print('Durchschnittlicher MAE NN aus 10 Durchläufen: ', mae_nn2)
print("relative Abweichung MAE LR: {:.2%} ".format(rel_mae_lr))
print("relative Abweichung MAE NN: {:.2%} ".format(rel_mae_nn))

""" ----- Plot zum Vergleich der Methoden ------- """
datenbeschriftung =['MAE LR','MAE NN']
data = [mae_lr2, mae_nn2]
s = pd.Series(data, index=datenbeschriftung)
s.plot(kind="bar", rot=0)
plt.xlabel('MAE')
plt.plot()
plt.show()
