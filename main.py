import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Fetching data using yfinance for a broader range (2010 to present)
df = yf.download('AAPL', start='2010-01-01', end='2024-09-24')

# Display the DataFrame
print(df.head())

# Plot
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# Prepare the dataset
data = df.filter(['Close'])
dataSet = data.values
training_data_len = math.ceil(len(dataSet) * 0.8)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataSet)

# Create the training data set
train_data = scaled_data[0:training_data_len, :]

# Split the data into x and y training datasets
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

# Convert the x and y training datasets to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=10)

# Testing data set
test_data = scaled_data[training_data_len - 60:, :]

x_test = []
y_test = dataSet[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculate the root mean squared error (RMSE)
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print(f'Root Mean Squared Error: {rmse}')

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid.loc[:, 'Predictions'] = np.concatenate(predictions)

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model Prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
plt.show()

# Show the valid/predicted prices
print(valid)

# Get the last 60 days closing prices and predict the next day
last_60_days = data[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)

# Prepare final data for prediction
X_test_last = np.reshape(last_60_days_scaled, (1, last_60_days_scaled.shape[0], 1))

# Get the predicted scaled price
pred_price = model.predict(X_test_last)

# Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(f'Predicted Close Price for the next day: ${pred_price[0][0]:.2f}')