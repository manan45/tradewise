from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Index(Base):
    __tablename__ = 'indexes'
    id = Column(Integer, primary_key=True)
    name = Column(String)

class Stock(Base):
    __tablename__ = 'stocks'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)

class MutualFund(Base):
    __tablename__ = 'mutual_funds'
    id = Column(Integer, primary_key=True)
    name = Column(String)

class ChannelData(Base):
    __tablename__ = 'channel_data'
    id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey('stocks.id'))
    channel_type = Column(String)
    timestamp = Column(DateTime)
    value = Column(Float)
    stock = relationship("Stock", back_populates="channels")

Stock.channels = relationship("ChannelData", order_by=ChannelData.id, back_populates="stock")
