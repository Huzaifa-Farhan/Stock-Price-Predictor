import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Fetching data using yfinance
df = yf.download('AAPL', start='2020-01-01', end='2024-09-23')

# Display the DataFrame
print(df)

#plot
df.shape
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date, fontsize = 18')
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.show()

#dataframe
data = df.filter(['Close'])
dataSet = data.values
training_data_len = math.ceil( len(dataSet) * .8)

