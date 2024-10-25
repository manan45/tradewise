from prophet import Prophet
import pandas as pd
from keras.models import Sequential
import numpy as np
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

def forecast_timeseries(data):
    df = pd.DataFrame(data)
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

def lstm_forecast(data, look_back=1):
    # Prepare data
    data = data[['y']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Create dataset
    def create_dataset(data, look_back=1):
        X, Y = [], []
        for i in range(len(data) - look_back - 1):
            a = data[i:(i + look_back), 0]
            X.append(a)
            Y.append(data[i + look_back, 0])
        return np.array(X), np.array(Y)

    X, Y = create_dataset(data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=10, batch_size=1, verbose=2)

    # Make predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions
