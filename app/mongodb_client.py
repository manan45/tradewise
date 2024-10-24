from pymongo import MongoClient
import os

def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_uri, maxPoolSize=200, wTimeoutMS=2500)
    return client

def get_database(db_name):
    client = get_mongo_client()
    return client[db_name]
