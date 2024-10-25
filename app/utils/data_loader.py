import pandas as pd
import os

def load_stock_data() -> pd.DataFrame:
    """
    Load sample stock data for testing and development.

    :return: DataFrame of stock data.
    """
    # Create a sample dataset
    data = {
        'date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
        'open': [100 + i * 0.1 for i in range(365)],
        'high': [101 + i * 0.1 for i in range(365)],
        'low': [99 + i * 0.1 for i in range(365)],
        'close': [100.5 + i * 0.1 for i in range(365)],
        'volume': [1000000 + i * 1000 for i in range(365)]
    }
    
    df = pd.DataFrame(data)
    
    # Add some random news sentiment
    df['news_sentiment'] = pd.Series([-1, 0, 1]).sample(n=365, replace=True).values
    
    return df

def get_latest_stock_data() -> pd.DataFrame:
    """
    Get the latest stock data for real-time processing.

    :return: DataFrame of the latest stock data.
    """
    df = load_stock_data()
    return df.tail(1)
