# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

pip install yfinance pandas numpy scikit-learn matplotlib

# +
import yfinance as yf
import pandas as pd

# Download historical data for AAPL
ticker_symbol = "AAPL"
aapl_data = yf.download(ticker_symbol, start="2010-01-01", end="2023-12-31")
# Save the DataFrame to a CSV file in data folder
aapl_data.to_csv('data/AAPL_stock_data.csv')
# -

aapl_data.head()

aapl_data.info()



# Check for missing values
aapl_data.isnull().sum()

# Fill missing values if any
aapl_data.fillna(method='ffill', inplace=True)  # Forward fill


# Example: Calculate a 5-day moving average
aapl_data['5_day_avg'] = aapl_data['Close'].rolling(window=5).mean()
# Fill NaN values with the first non-NaN average or another value
aapl_data['5_day_avg'].fillna(method='bfill', inplace=True)  # Backward fill
# or
aapl_data['5_day_avg'].fillna(aapl_data['Close'], inplace=True)  # Fill with the closing price


aapl_data.head()

# Time series visualization
import matplotlib.pyplot as plt

aapl_data['Close'].plot(figsize=(10, 5))
plt.title('AAPL Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()

# -

# Volume Analysis
aapl_data['Volume'].plot(figsize=(10, 5))
plt.title('AAPL Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()

# +
# Check for Stationary
from statsmodels.tsa.stattools import adfuller

result = adfuller(aapl_data['Close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# +
# Moving Average
aapl_data['MA_20'] = aapl_data['Close'].rolling(window=20).mean()
aapl_data['MA_50'] = aapl_data['Close'].rolling(window=50).mean()

aapl_data[['Close', 'MA_20', 'MA_50']].plot(figsize=(10, 5))
plt.title('AAPL Close Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
# -

correlation = aapl_data.corr()
correlation

# +
# Seasonal Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(aapl_data['Close'], model='multiplicative', period=365)
result.plot()
plt.show()


# +
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Prepare features and target
X = aapl_data[['Open', 'High', 'Low', '5_day_avg']]
y = aapl_data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# -

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# +
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Predict and evaluate
predictions = model.predict(X_test)
mse_lr = mean_squared_error(y_test, predictions)
mae_lr = mean_absolute_error(y_test, predictions)
mse_lr, mae_lr

# +
from sklearn.ensemble import RandomForestRegressor

# Create and train the model
rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(X_train, y_train)
# -

# Predict and evaluate
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse, rf_mae

pip install tensorflow

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(aapl_data)


# Create dataset for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]   
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


time_step = 100
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1) # Reshape for LSTM

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=[MeanSquaredError(), MeanAbsoluteError()])
lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# Predict and evaluate (you may need to inverse transform to get actual values)
lstm_predictions = lstm_model.predict(X_test)

from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
# Calculate MSE and MAE
lstm_mse = mean_squared_error(y_test, lstm_predictions)
lstm_mae = mean_absolute_error(y_test, lstm_predictions)
lstm_mse, lstm_mae

# +
import matplotlib.pyplot as plt

# Assuming you have these values from your models' evaluations
mse_values = {
    'Linear Regression': mse_lr,  # Replace with your actual MSE value
    'Random Forest': rf_mse,          # Replace with your actual MSE value
    'LSTM': lstm_mse                  # Replace with your actual MSE value
}

mae_values = {
    'Linear Regression': mae_lr,  # Replace with your actual MAE value
    'Random Forest': rf_mae,          # Replace with your actual MAE value
    'LSTM': lstm_mae                  # Replace with your actual MAE value
}

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# MSE Comparison
ax[0].bar(mse_values.keys(), mse_values.values(), color='blue')
ax[0].set_title('MSE Comparison')
ax[0].set_ylabel('Mean Squared Error')
ax[0].set_xlabel('Model')

# MAE Comparison
ax[1].bar(mae_values.keys(), mae_values.values(), color='green')
ax[1].set_title('MAE Comparison')
ax[1].set_ylabel('Mean Absolute Error')
ax[1].set_xlabel('Model')

plt.tight_layout()
plt.show()

# +
# This will create inputs X and targets y
X, y = create_dataset(scaled_data)

# Reshape the input data to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# +

# Initialize the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
# -

# Compile and fit the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, batch_size=32)

# +
# Save the model
model.save('lstm_model.h5')

# Save the scaler
import joblib
scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename)
# -

# Inverse scaling (if your data was scaled)
y_pred = model.predict(X)



import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(actual_prices_flattened, label='Actual Prices', color='blue')
plt.plot(predicted_prices_flattened, label='Predicted Prices', color='red')
plt.title('Comparison of Actual and Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


y_pred

# +
min_val = scaler.min_[0]
scale_val = scaler.scale_[0]
original_min_val = scaler.data_min_[0]
original_max_val = scaler.data_max_[0]


actual_prices = y / scale_val + min_val
predicted_prices = y_pred / scale_val + min_val

actual_prices_flattened = actual_prices.flatten() if actual_prices.ndim > 1 else actual_prices
predicted_prices_flattened = predicted_prices.flatten() if predicted_prices.ndim > 1 else predicted_prices

# -

import pandas as pd
AAPL_predictions = {
    'Actual Price': actual_prices_flattened,
    'Predicted Price': predicted_prices_flattened
}
df = pd.DataFrame(AAPL_predictions)


df.head()

# +
csv_filename = 'stock_prices_predictions.csv'
df.to_csv(csv_filename, index=False)

print(f"Data exported to {csv_filename}")
# -


