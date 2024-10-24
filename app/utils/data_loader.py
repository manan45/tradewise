import pandas as pd
import os

def load_stock_data_in_chunks(chunk_size=100000) -> pd.DataFrame:
    """
    Load stock data in chunks for efficient processing.

    :param chunk_size: Number of rows per chunk.
    :return: DataFrame of stock data.
    """
    file_path = os.path.join(os.path.dirname(__file__), 'stock_data.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")
    
    # Load data in chunks
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    df = pd.concat(chunks)
    
    # Convert all columns to lower case
    df.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('₹', 'inr').rstrip('_') for col in df.columns]
    
    # Convert 'date' column to datetime format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
    else:
        raise KeyError("The required column 'date' is not present in the data.")
    
    return df

