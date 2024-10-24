# app/core/entities/stock.py

class Stock:
    def __init__(self, symbol: str, price: float):
        self.symbol = symbol
        self.price = price

    def update_price(self, new_price: float):
        self.price = new_price