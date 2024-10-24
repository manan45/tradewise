import requests
import pandas as pd
from datetime import datetime

def fetch_data_from_api(api_url):
    """
    Fetch data from a given API URL.

    :param api_url: The API endpoint URL.
    :return: DataFrame containing the fetched data.
    """
    response = requests.get(api_url)
    data = response.json()
    df = pd.DataFrame(data)
    return df

def fetch_real_time_data():
    """
    Fetch real-time data from multiple sources.

    :return: Combined DataFrame of all data sources.
    """
    # Example API URLs
    news_api_url = "https://api.example.com/news"
    oi_api_url = "https://api.example.com/open_interest"
    
    news_data = fetch_data_from_api(news_api_url)
    oi_data = fetch_data_from_api(oi_api_url)
    
    # Combine data
    combined_data = pd.concat([news_data, oi_data], axis=1)
    combined_data['timestamp'] = datetime.now()
    return combined_data
