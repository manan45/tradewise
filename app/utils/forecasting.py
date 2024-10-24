from prophet import Prophet
import pandas as pd

def forecast_timeseries(data):
    df = pd.DataFrame(data)
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]
