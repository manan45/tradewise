# Dhan API driver implementation

class DhanAPI:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_data(self, endpoint):
        # Implement data fetching logic using the provided endpoint
        pass
class DhanAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        # Initialize API client here

    def fetch_data(self, endpoint):
        # Code to fetch data from the API
        # This should include making an HTTP request to the API endpoint
        response = requests.get(f"https://api.dhan.com{endpoint}", headers={"Authorization": f"Bearer {self.api_key}"})
        return response.json()

    def fetch_historical_data(self, symbol, start_date, end_date):
        endpoint = f"/historical/{symbol}?start={start_date}&end={end_date}"
        return self.fetch_data(endpoint)
