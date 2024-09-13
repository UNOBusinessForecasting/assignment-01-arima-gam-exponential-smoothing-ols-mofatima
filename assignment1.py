
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pickle  # For sa
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


train_data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv')
train_data.head()
test_data = pd.read_csv('https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv')
test_data.head()

#Convert the timestamp column to datetime 
train_data['Timestamp'] = pd.to_datetime(train_data['Timestamp'])
#Set the timestamp as the index for the time series
train_data.set_index('Timestamp', inplace=True)
print(train_data.head())

#Plot to check for seasonality or trends 
train_data['trips'].plot()

#check for missing or na values
train_data.isnull().sum()
train_data.isna().sum()

 # Check for stationarity with rolling statistics or ADF test
result = adfuller(train_data['trips'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}') 

#Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(16,6))
plot_acf(train_data['trips'], lags=50, ax=axes[0])
axes[0].set_title('ACF Plot')
plot_pacf(train_data['trips'], lags=50, ax=axes[1])
axes[1].set_title('PACF Plot')
plt.show()

#Fit an ARIMA model
p, d, q = 2, 0, 1
model = ARIMA(train_data['trips'], order=(p, d, q))
modelFit = model.fit()

#Print the model summary
print(modelFit.summary())

#Forecast for the test period (744 hours)
pred = modelFit.forecast(steps=744)
pred

#Evalution of model
test_data['preicted_mean']= pred
test_data 

#Save predictions
pred = pd.DataFrame(pred, columns=['forecast'])
pred.to_csv('predictions.csv', index=False)

#Save the model to a file
with open('modelFit.pkl', 'wb') as pkl_file:
    pickle.dump(modelFit, pkl_file)


