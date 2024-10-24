import pandas as pd
import os
import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
import os
import numpy as np

def load_stock_data() -> pd.DataFrame:
    file_path = os.path.join(os.path.dirname(__file__), 'stock_data.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")
    # Load data from CSV file
    df = pd.read_csv(file_path)
    # Convert all columns to lower case
    df.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('₹', 'inr').rstrip('_') for col in df.columns]
    # Convert 'date' column to datetime format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
    else:
        raise KeyError("The required column 'date' is not present in the data.")
    return df

