# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:35:43 2018
multiple linear regression
@author: AQ44828
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

# load data set 
boston = datasets.load_boston(return_X_y = False)
X = boston['data']
y = boston['target']

# split the data into test and train
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1)

# Multiple linear regression
reg = linear_model.LinearRegression()

# fit the regression model on train data 
reg.fit(x_train, y_train)

# co-efficients of features
print("coefficient value",reg.coef_)

#variance score on test set
print("variance score", reg.score(x_test,y_test))

# variance score on train set
print("variance score on train set:" , reg.score(x_train, y_train))

# residual plots
plt.style.use('fivethirtyeight')

plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train, color = "green", s =10, label = "Training Data")

plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test, color = "blue", s = 10, label = "Test Data")

plt.hlines( y =0, xmin= 0, xmax= 50, linewidth =2)

plt.legend( loc = "upper right")

plt.title('Residual plot')

plt.show()