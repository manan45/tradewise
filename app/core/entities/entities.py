from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Index(Base):
    __tablename__ = 'indexes'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

class Stock(Base):
    __tablename__ = 'stocks'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, unique=True)

class MutualFund(Base):
    __tablename__ = 'mutual_funds'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

class ChannelData(Base):
    __tablename__ = 'channel_data'
    id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    channel_type = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    value = Column(Float, nullable=False)
    stock = relationship("Stock", back_populates="channels")

Stock.channels = relationship("ChannelData", order_by=ChannelData.id, back_populates="stock")
