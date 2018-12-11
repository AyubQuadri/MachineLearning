# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:01:36 2018
https://www.geeksforgeeks.org/linear-regression-python-implementation/
@author: AQ44828
"""

# simple linar regression using python pandas
import numpy as np
import matplotlib.pyplot as plt


def estimate_coef(x,y):
    n = np.size(x)
    
    m_x, m_y = np.mean(x), np.mean(y)
    
    # calculate the cross deviation and deviation about x
    SS_xy = np.sum(y*x - n * m_y * m_x)
    SS_xx = np.sum(x*x - n * x*x)
    
    b_1 = SS_xy/SS_xx
    b_0 = m_y - b_1 * m_x
    
    return (b_1, b_0)

def plot_regression_line(x,y,b):
    plt.scatter(x,y, color = "m", marker = "o", s = 30)
    
    y_pred = b[0]+b[1]*x
    
    plt.plot(x, y_pred, color ='g')
    
    plt.xlabel('X-Values')
    plt.ylabel('Y-values')
    
    plt.show()
    return

def main():
    
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([1,3,2,5,7,8,8,9,10,12])
    
    b = estimate_coef(x,y)
    print('estimated Coef values \n b_0 = {} \n b_1 = {}'.format(b[0],b[1]) )
    
    plot_regression_line(x,y,b)
    
if __name__ == "__main__":
    main()
    
    