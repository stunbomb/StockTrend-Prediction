import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error

#Data retrieved from Yahoo finance
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values 

training_set
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
x_train = []
y_train = []
for i in range(120, 1258): #we have 1258 dates in our excel file
    x_train.append(training_set_scaled[i-120:i, 0])
    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train),np.array(y_train)
np.savetxt("predicted_x_train.csv", x_train, delimiter=",")
np.savetxt("predicted_y_train.csv", y_train, delimiter=",")

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units = 128, activation='tanh', return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=128, activation='tanh',return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=128, activation='tanh',return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=128, activation='tanh',return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(x_train, y_train, epochs=100, batch_size=32)
model.summary()

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(120, 140):
    X_test.append(inputs[i-120:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

np.savetxt("predicted_Forex_price.csv", predicted_stock_price, delimiter=",")
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print(rmse/800)