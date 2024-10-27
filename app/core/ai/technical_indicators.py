import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import AccDistIndexIndicator, OnBalanceVolumeIndicator, ForceIndexIndicator
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator
import warnings

class TechnicalIndicatorCalculator:
    """Advanced technical indicator calculator with comprehensive market analysis"""
    
    def __init__(self):
        self.epsilon = 1e-8  # Small value to avoid division by zero

    def safe_divide(self, a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators for market analysis
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated technical indicators
        """
        df = self._clean_dataframe(df)
        
        df = self._calculate_price_action_indicators(df)
        df = self._calculate_trend_indicators(df)
        df = self._calculate_momentum_indicators(df)
        df = self._calculate_volatility_indicators(df)
        df = self._calculate_volume_indicators(df)
        df = self._calculate_support_resistance(df)
        df = self._calculate_fibonacci_levels(df)
        df = self._calculate_composite_indicators(df)
        
        return df

    def _calculate_price_action_indicators(self, df: pd.DataFrame):
        """Calculate price action based indicators"""
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        df['is_doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'])) < 0.1
        return df

    def _calculate_trend_indicators(self, df: pd.DataFrame):
        """Calculate trend-following indicators"""
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        ema_20 = EMAIndicator(close=df['close'], window=20)
        ema_50 = EMAIndicator(close=df['close'], window=50)
        df['ema_20'] = ema_20.ema_indicator()
        df['ema_50'] = ema_50.ema_indicator()
        
        sma_200 = SMAIndicator(close=df['close'], window=200)
        df['sma_200'] = sma_200.sma_indicator()
        
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
        df['adx'] = adx.adx()
        
        return df

    def _calculate_momentum_indicators(self, df: pd.DataFrame):
        """Calculate momentum-based indicators"""
        rsi = RSIIndicator(close=df['close'])
        df['rsi'] = rsi.rsi()
        
        stoch_rsi = StochRSIIndicator(close=df['close'])
        df['stoch_rsi'] = stoch_rsi.stochrsi()
        
        williams_r = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'])
        df['williams_r'] = williams_r.williams_r()
        
        return df

    def _calculate_volatility_indicators(self, df: pd.DataFrame):
        """Calculate volatility indicators"""
        bb = BollingerBands(close=df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
        df['atr'] = atr.average_true_range()
        
        kc = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'])
        df['kc_high'] = kc.keltner_channel_hband()
        df['kc_low'] = kc.keltner_channel_lband()
        df['kc_mid'] = kc.keltner_channel_mband()
        
        return df

    def _calculate_volume_indicators(self, df: pd.DataFrame):
        """Calculate volume-based indicators"""
        adi = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
        df['adi'] = adi.acc_dist_index()
        
        obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = obv.on_balance_volume()
        
        fi = ForceIndexIndicator(close=df['close'], volume=df['volume'])
        df['force_index'] = fi.force_index()
        
        return df

    def _calculate_support_resistance(self, df: pd.DataFrame):
        """Calculate support and resistance levels"""
        window = 20
        df['rolling_high'] = df['high'].rolling(window=window).max()
        df['rolling_low'] = df['low'].rolling(window=window).min()
        
        df['support'] = df['rolling_low'].where(df['rolling_low'] < df['close'], np.nan)
        df['resistance'] = df['rolling_high'].where(df['rolling_high'] > df['close'], np.nan)
        
        return df

    def _calculate_fibonacci_levels(self, df: pd.DataFrame):
        """Calculate Fibonacci retracement levels"""
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low
        
        df['fib_0'] = low
        df['fib_23.6'] = low + 0.236 * diff
        df['fib_38.2'] = low + 0.382 * diff
        df['fib_50'] = low + 0.5 * diff
        df['fib_61.8'] = low + 0.618 * diff
        df['fib_100'] = high
        
        return df

    def _calculate_composite_indicators(self, df: pd.DataFrame):
        """Calculate custom composite indicators"""
        df['trend_strength'] = (df['adx'] + df['rsi']) / 2
        df['volatility_index'] = (df['atr'] / df['close']) * 100
        df['momentum_score'] = (df['rsi'] + (100 + df['williams_r']) / 2) / 2
        
        return df

    def _clean_dataframe(self, df: pd.DataFrame):
        """Clean and validate the dataframe"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df = df.dropna()
        df = df.sort_index()
        
        return df
