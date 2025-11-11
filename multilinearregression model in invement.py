import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\DELL\Downloads\Investment.csv")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:,4]

#change categrocial to numerical data
x = pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#fiting the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

bias=regressor.score(x_train,y_train)
bias

variance = regressor.score(x_test,y_test)
variance

slope=regressor.coef_
print(slope)

intercept= regressor.intercept_
print(intercept)

#add constant to data set
x = np.append(arr=np.ones((50,1)).astype(int), values = x, axis=1)

import statsmodels.api as sm
x_otp= x[:,[0,1,2,3,4,5]]
#ordinaryleastsquares endog=input , exog=output
regressor_OLS = sm.OLS(endog=y, exog=x_otp).fit()
regressor_OLS.summary() 

#rfe=recursive feature elimination
import statsmodels.api as sm
x_otp= x[:,[0,1,2,3,5]]
#ordinaryleastsquares endog=input , exog=output
regressor_OLS = sm.OLS(endog=y, exog=x_otp).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_otp= x[:,[0,1,2,3]]
#ordinaryleastsquares endog=input , exog=output
regressor_OLS = sm.OLS(endog=y, exog=x_otp).fit()
regressor_OLS.summary() 

import statsmodels.api as sm
x_otp= x[:,[0,1,2]]
#ordinaryleastsquares endog=input , exog=output
regressor_OLS = sm.OLS(endog=y, exog=x_otp).fit()
regressor_OLS.summary() 

import statsmodels.api as sm
x_otp= x[:,[0,1,]]
#ordinaryleastsquares endog=input , exog=output
regressor_OLS = sm.OLS(endog=y, exog=x_otp).fit()
regressor_OLS.summary() 





























