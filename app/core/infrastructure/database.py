# Database connection and configuration

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class Database:
    def __init__(self, uri):
        self.engine = create_engine(uri)
        self.Session = sessionmaker(bind=self.engine)

    def connect(self):
        return self.Session()
