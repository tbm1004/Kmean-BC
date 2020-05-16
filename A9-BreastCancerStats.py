# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 05:10:58 2018

@author: tbm1004
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

#from sklearn.cross_validation import train_test_split
#new version of sklearn has train_test_split in model_selection library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn import metrics  

#fill up a dataframe with data

def mregress(file,inpx,inpy):
    dataset = pd.read_csv(file)  
    print(dataset.shape)
    print(dataset.head())
    print(dataset.describe())
    
    #prepare the data
    #create attributes and labels
    #use column names for creating an attribute set and label
    x = dataset[inpx]
    y = dataset[inpy]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    regressor = LinearRegression()  
    regressor.fit(x_train, y_train) 
    
    print()
    #the regression model has to find the most optimal coefficients for all the attributes
    coeff_df = pd.DataFrame(regressor.coef_, x.columns, columns=['Coefficient'])  
    print(coeff_df)
    
    y_pred = regressor.predict(x_test)  
    print()
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
    print(df)
    print()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def define():
    print("Pima CSV:")
    pimax = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigree','Age']
    pimay = 'Outcome'
    mregress('pimasmall.csv', pimax, pimay)
    print()
    print("Breast Cancer CSV:")
    bcx = ['thickness','cellsize','cellshape','adhesion','epithelial','neclei','chromatin','nucleoli','mitoses']
    bcy = 'class'
    mregress('breastcancer.csv', bcx, bcy)
    
define()
    
    
