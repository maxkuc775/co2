# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataSet = pd.read_csv("Automobile_data.txt",
                      delimiter = ',',
                      thousands = None,
                      decimal = '.')

dataSet_rpm = dataSet[dataSet['peak-rpm'] != '?'][['engine-type', 'num-of-cylinders','peak-rpm']]
dataSet_rpm['peak-rpm'] = dataSet_rpm['peak-rpm'].astype(float)
dataSet_rpm_grouped = dataSet_rpm.groupby(['engine-type', 'num-of-cylinders'])

dataSet_rpm_grouped['peak-rpm'].mean()

dataSet_c = dataSet.copy()

dataSet_c.loc[dataSet_c['peak-rpm'] == '?', 'peak-rpm'] = 5700


dataSet_c = dataSet_c[dataSet_c['price'] != '?']
dataSet_c['price'] = dataSet_c['price'].astype(float)


dataSet_c = dataSet_c[dataSet_c.horsepower != '?']
dataSet_c['horsepower'] = dataSet_c['horsepower'].astype(float)

""" ----- Vorbereitung für die Regressionsanalyse ----- """

cols_ratio = ['horsepower', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'compression-ratio', 'city-mpg', 'highway-mpg']
cols_target = ['price']

dataSet_ratio = dataSet_c.loc[:, cols_ratio]
dataSet_target = dataSet_c[cols_target]

"""
grr = pd.plotting.scatter_matrix(dataSet_c[cols_target + cols_ratio]
                                 ,c = dataSet_target
                                 ,figsize=(15, 15)
                                 ,marker = 'o'
                                 ,hist_kwds={'bins' : 20}
                                 ,s = 60
                                 ,alpha = 0.8)
plt.show()
"""

X = dataSet_ratio[['highway-mpg']]
y = dataSet_c[cols_target]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = None)

""" ----- Lineare Regressionsanalyse ------- """

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)


print('------ Lineare Regression -----')
print('Funktion via sklearn: y = %.3f * x + %.3f' % (lr.coef_[0], lr.intercept_))
print("Alpha: {}".format(lr.intercept_))
print("Beta: {}".format(lr.coef_[0]))
print("Training Set R² Score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test Set R² Score: {:.2f}".format(lr.score(X_test, y_test)))
print("\n")

""" ----- Polynomial Regressionsanalyse ------- """
from sklearn.preprocessing import PolynomialFeatures

quadratic = PolynomialFeatures(degree = 3)

X_train_quad = quadratic.fit_transform(X_train)
X_test_quad = quadratic.fit_transform(X_test)

pr = LinearRegression()

pr.fit(X_train_quad, y_train)

X_fit = np.arange(X_train.min().values, X_train.max().values)
X_fit = X_fit[:, np.newaxis]

X_fit_quad = quadratic.fit_transform(X_fit)

""" ------- Decision Tree Regressor ------ """

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=5)

tree.fit(X_train, y_train)


plt.figure(figsize=(10,10))
plt.scatter(X_train, y_train, color = 'blue')
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, lr.predict(X_train), color = 'red', linewidth = 2.0)
plt.plot(X_fit, pr.predict(X_fit_quad), color = 'purple', linewidth = 2.0)
plt.plot(X_fit, tree.predict(X_fit), color = 'orange', linewidth = 2.0)
plt.xlabel(X_train.columns[0])
plt.ylabel(cols_target[0])

plt.show()



