import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_prediction_model(data):
    """
    Train a prediction model using RandomForestRegressor.

    :param data: DataFrame containing features and target variable.
    :return: Trained model.
    """
    # Example feature engineering
    data['lag_1'] = data['close'].shift(1)
    data['lag_2'] = data['close'].shift(2)
    data.dropna(inplace=True)

    X = data[['lag_1', 'lag_2', 'sentiment_score']]
    y = data['close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error: {mse}")

    return model

def generate_predictions(model, data):
    """
    Generate predictions for the next 2 days in 15-minute intervals.

    :param model: Trained prediction model.
    :param data: DataFrame containing features for prediction.
    :return: DataFrame with predictions.
    """
    data['predictions'] = model.predict(data[['lag_1', 'lag_2', 'sentiment_score']])
    return data
