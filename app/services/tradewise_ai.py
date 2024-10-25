import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import gym
from gym import spaces
from transformers import pipeline
from prophet import Prophet
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.total_reward = 0
        self.holdings = 0
        self.cash = 10000  # Starting cash
        return self._next_observation()

    def _next_observation(self):
        obs = np.array([
            self.data['close'].iloc[self.current_step],
            self.data['macd'].iloc[self.current_step],
            self.data['rsi'].iloc[self.current_step],
            self.data['bb_high'].iloc[self.current_step],
            self.data['bb_low'].iloc[self.current_step]
        ])
        return obs

    def step(self, action):
        self.current_step += 1
        current_price = self.data['close'].iloc[self.current_step]
        reward = 0

        if action == 1:  # Buy
            shares_to_buy = self.cash // current_price
            self.holdings += shares_to_buy
            self.cash -= shares_to_buy * current_price
        elif action == 2:  # Sell
            reward = self.holdings * current_price - self.holdings * self.data['close'].iloc[self.current_step - 1]
            self.cash += self.holdings * current_price
            self.holdings = 0

        done = self.current_step >= len(self.data) - 1
        obs = self._next_observation()
        return obs, reward, done, {}

class TradewiseAI:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")

    def create_env(self, data):
        return StockTradingEnv(data)

    def train_agent(self, env, n_estimators=100, random_state=42):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        training_data = []
        training_labels = []

        for _ in range(1000):  # Run 1000 episodes
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()  # Random action for data collection
                next_obs, reward, done, _ = env.step(action)
                training_data.append(obs)
                training_labels.append(reward)
                obs = next_obs

        model.fit(training_data, training_labels)
        return model

    def generate_trade_suggestions(self, data: pd.DataFrame) -> list:
        # Prepare features
        macd = MACD(close=data['close'])
        rsi = RSIIndicator(close=data['close'])
        bb = BollingerBands(close=data['close'])

        data['macd'] = macd.macd()
        data['rsi'] = rsi.rsi()
        data['bb_high'] = bb.bollinger_hband()
        data['bb_low'] = bb.bollinger_lband()

        env = self.create_env(data)
        model = self.train_agent(env)

        suggestions = []
        obs = env.reset()
        for _ in range(5):  # Generate suggestions for the next 5 steps
            action = model.predict([obs])[0]
            if action == 1:
                suggestions.append({"action": "BUY", "price": data['close'].iloc[env.current_step], "confidence": 0.8})
            elif action == 2:
                suggestions.append({"action": "SELL", "price": data['close'].iloc[env.current_step], "confidence": 0.8})
            obs, _, done, _ = env.step(action)
            if done:
                break

        return suggestions

    def train_prediction_model(self, data):
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

    def generate_predictions(self, model, data):
        data['predictions'] = model.predict(data[['lag_1', 'lag_2', 'sentiment_score']])
        return data

    def analyze_sentiment(self, news_articles):
        sentiments = [self.sentiment_pipeline(article)[0] for article in news_articles]
        return sentiments

    def get_sentiment_scores(self, data):
        data['sentiment_score'] = self.analyze_sentiment(data['news'])
        return data

    def forecast_timeseries(self, data):
        df = pd.DataFrame(data)
        df.columns = ['ds', 'y']
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat']]

    def lstm_forecast(self, data, look_back=1):
        data = data[['y']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)

        def create_dataset(data, look_back=1):
            X, Y = [], []
            for i in range(len(data) - look_back - 1):
                a = data[i:(i + look_back), 0]
                X.append(a)
                Y.append(data[i + look_back, 0])
            return np.array(X), np.array(Y)

        X, Y = create_dataset(data, look_back)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, Y, epochs=10, batch_size=1, verbose=2)

        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        return predictions

    def calculate_moving_average(self, data: pd.Series, window: int) -> pd.Series:
        """
        Calculate the simple moving average for a given window.

        :param data: Pandas Series containing the price data
        :param window: The number of periods to use for the moving average
        :return: Pandas Series containing the moving average values
        """
        return data.rolling(window=window).mean()

    def calculate_exponential_moving_average(self, data: pd.Series, window: int) -> pd.Series:
        """
        Calculate the exponential moving average for a given window.

        :param data: Pandas Series containing the price data
        :param window: The number of periods to use for the exponential moving average
        :return: Pandas Series containing the exponential moving average values
        """
        return data.ewm(span=window, adjust=False).mean()

def main():
    ai = TradewiseAI()
    # Add example usage here

if __name__ == "__main__":
    main()
