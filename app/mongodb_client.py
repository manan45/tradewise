from pymongo import MongoClient
import os

def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_uri, maxPoolSize=500, wTimeoutMS=2500)
    return client
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_uri, maxPoolSize=500, wTimeoutMS=2500)
    return client
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_uri, maxPoolSize=200, wTimeoutMS=2500)
    return client

def get_database(db_name):
    client = get_mongo_client()
    return client[db_name]
from pymongo import MongoClient

class MongoDBClient:
    def __init__(self, uri):
        self.client = MongoClient(uri)
        self.db = self.client['stock_database']

    def insert_stock_data(self, stock_data):
        self.db.stocks.insert_one(stock_data)

    def get_all_stocks(self):
        return list(self.db.stocks.find())
