import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def load_data():
    df = pd.read_csv("Weather_dataset.csv")
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

def smooth_data(data, window_size=3):
    return data.rolling(window=window_size).mean()

def train_model_1():
    df = load_data()
    df['temperature_smoothed'] = smooth_data(df['temperature'])

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['temperature_smoothed']].dropna())

    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)

    sequence_length = 10
    X, y = create_sequences(df_scaled, sequence_length)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
    
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions)

    mse = mean_squared_error(scaler.inverse_transform(y_test), predictions_rescaled)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-len(predictions):], scaler.inverse_transform(y_test), label="Gerçek Değerler")
    plt.plot(df.index[-len(predictions):], predictions_rescaled, label="LSTM Tahminleri", linestyle='dashed')
    plt.xlabel("Zaman")
    plt.ylabel("Sıcaklık")
    plt.legend()
    plt.title("LSTM ile Tahminleme")

    plot_path = 'static/lstm_temperature_forecast.png'
    plt.savefig(plot_path)
    plt.close()

    return mse, plot_path
