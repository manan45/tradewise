# MongoDB client implementation

from pymongo import MongoClient

class MongoDBClient:
    def __init__(self, uri):
        self.client = MongoClient(uri)

    def connect(self, db_name):
        return self.client[db_name]
