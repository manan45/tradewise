from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.entities.entities import Base, Stock, ChannelData

engine = create_engine('postgresql://user:password@localhost/mydatabase')
Session = sessionmaker(bind=engine)
session = Session()

def ingest_data(data):
    for item in data:
        stock = Stock(symbol=item['symbol'])
        session.add(stock)
        for channel, value in item['channels'].items():
            channel_data = ChannelData(entity_id=stock.id, channel_type=channel, value=value)
            session.add(channel_data)
    session.commit()
